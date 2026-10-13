import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from clasificador_tickets import (limpiar_texto, normalizar_categoria, preparar_datos,
    dividir_datos, crear_pipeline, predecir_textos, clasificar_texto, guardar_csv,
    entrenar, cargar_modelo, leer_tickets)


def fila(numero, texto, categoria, ticket=None):
    return {"fila_excel": numero, "ticket": ticket or str(numero), "texto_q": texto,
            "categoria_ab": categoria, "texto": limpiar_texto(texto),
            "categoria": normalizar_categoria(categoria)}


class ClasificadorTests(unittest.TestCase):
    def test_normalizacion_no_inventa_etiquetas(self):
        self.assertEqual(limpiar_texto("  CAMBIÓ\n válvula!!! "), "cambio valvula")
        self.assertEqual(limpiar_texto(None), "")
        self.assertEqual(limpiar_texto(float("nan")), "")
        self.assertEqual(normalizar_categoria("  IO   BOard "), "io board")
        self.assertNotEqual(normalizar_categoria("Pump alarm"), normalizar_categoria("Pump"))

    def test_excluir_contradicciones_y_duplicados(self):
        filas = [fila(2, "mismo texto", "Motor"), fila(3, "MISMO TEXTO", "motor"),
                 fila(4, "texto ambiguo", "motor"), fila(5, "texto ambiguo", "Pump"),
                 fila(6, "", "Pump"), fila(7, "texto solo", "")]
        datos, conflictos, nombres, auditoria = preparar_datos(filas)
        self.assertEqual(len(datos), 1)
        self.assertEqual(len(conflictos), 2)
        self.assertEqual(auditoria["repeticiones_texto_categoria_excluidas"], 1)
        self.assertEqual(nombres["motor"], "motor")

    def test_grupos_transitivos_y_separacion_sin_fuga(self):
        filas = [fila(2, "texto compartido", "motor", "T1"),
                 fila(3, "otro texto", "motor", "T1"),
                 fila(4, "otro texto", "motor", "T2"),
                 fila(5, "tercer texto", "motor", "T2")]
        filas += [fila(10+i, f"texto independiente {i}", "pump" if i%2 else "motor") for i in range(20)]
        datos, _, _, _ = preparar_datos(filas)
        self.assertEqual(len({f["grupo"] for f in datos[:3]}), 1)
        train, test = dividir_datos(datos)
        for campo in ["grupo", "texto", "ticket"]:
            self.assertFalse({datos[i][campo] for i in train} & {datos[i][campo] for i in test})

    def test_vacios_y_vocabulario_desconocido(self):
        p = crear_pipeline().fit(["motor electrico roto", "reemplazo motor", "fuga aceite", "relleno aceite"],
                                 ["motor", "motor", "aceite", "aceite"])
        modelo = {"pipeline": p, "nombres": {"motor": "Motor", "aceite": "Aceite"},
                  "conteos": {"motor": 2, "aceite": 2}}
        resultados = predecir_textos([None, "  ", "!!!", "漢字漢字", "motor electrico roto"], modelo)
        self.assertTrue(all(r["categoria"] is None for r in resultados[:4]))
        self.assertEqual(resultados[3]["estado"], "sin_vocabulario_conocido")
        self.assertEqual(resultados[4]["categoria"], "Motor")
        self.assertEqual(len(resultados[4]["alternativas"]), 2)
        self.assertEqual(clasificar_texto("motor electrico roto", modelo), "Motor")
        self.assertEqual(predecir_textos([], modelo), [])
        with self.assertRaises(ValueError):
            predecir_textos(["motor"], modelo, lote=0)

    def test_modelo_persistido_y_evaluacion(self):
        filas = [fila(2+i, f"{texto} pieza {i}", categoria)
                 for i, (texto, categoria) in enumerate(
                     [("motor electrico reemplazo", "Motor"), ("fuga aceite tanque", "Oil"),
                      ("puerta seguridad ajuste", "Door")] * 12)]
        with tempfile.TemporaryDirectory() as temporal:
            base = Path(temporal)
            archivo = base / "fuente.xlsm"
            archivo.write_bytes(b"fuente sin modificar")
            destino = base / "modelo.joblib"
            with patch("clasificador_tickets.leer_tickets", return_value=filas):
                resultado = entrenar(archivo, modelo=destino, salida=base / "salida")
            recuperado = cargar_modelo(destino)
            self.assertEqual(clasificar_texto("motor electrico reemplazo", recuperado), "Motor")
            self.assertEqual(resultado["metricas"]["auditoria"]["ejemplos_utilizables"], 36)
            self.assertEqual(sum(recuperado["conteos"].values()), 36)
            self.assertEqual(archivo.read_bytes(), b"fuente sin modificar")
            self.assertTrue((base / "salida" / "evaluacion.csv").is_file())

    def test_csv_no_sobrescribe_libro_y_escapa_formulas(self):
        with tempfile.TemporaryDirectory() as temporal:
            salida = Path(temporal) / "salida.csv"
            guardar_csv(salida, [{"texto": "=1+1"}, {"texto": "normal"}], ["texto"])
            self.assertIn("'=1+1", salida.read_text(encoding="utf-8-sig"))
            with self.assertRaises(ValueError):
                guardar_csv(Path(temporal) / "original.xlsm", [], ["texto"])

    def test_datos_insuficientes(self):
        with self.assertRaises(ValueError):
            preparar_datos([fila(2, "", "motor")])
        datos, _, _, _ = preparar_datos([fila(2, "motor falla", "motor")])
        with self.assertRaises(ValueError):
            dividir_datos(datos)


if __name__ == "__main__":
    unittest.main()
