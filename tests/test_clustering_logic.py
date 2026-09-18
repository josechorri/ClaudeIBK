# -*- coding: utf-8 -*-
"""
Prueba (sin Spark) de la lógica de clustering/fallback de outliers_consolidado_clustering.py.

Importa SOLO la parte pura-python (config + funciones auxiliares + analizar_combinacion), stubbeando
pyspark, y ejercita el criterio de aceptación y los casos borde.

    python3 tests/test_clustering_logic.py

Requiere: numpy, pandas, scikit-learn (opcional; hay fallback numpy).
"""
import os, sys, types
import numpy as np
import pandas as pd

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "outliers_consolidado_clustering.py")

# --- Cargar solo el 'head' (hasta la ingesta Spark) para no ejecutar el pipeline ---
src = open(SCRIPT, encoding="utf-8").read()
head = src.split("#  1-3) INGESTA")[0]

# Stubs de pyspark para que el import de nivel superior no falle.
mod = types.ModuleType("pyspark"); sys.modules["pyspark"] = mod
sqlmod = types.ModuleType("pyspark.sql"); sys.modules["pyspark.sql"] = sqlmod
fmod = types.ModuleType("pyspark.sql.functions"); sys.modules["pyspark.sql.functions"] = fmod
tmod = types.ModuleType("pyspark.sql.types"); sys.modules["pyspark.sql.types"] = tmod
for name in ["StructType", "StructField", "StringType", "IntegerType", "LongType",
             "DoubleType", "BooleanType"]:
    setattr(tmod, name, lambda *a, **k: None)
sqlmod.functions = fmod; sqlmod.types = tmod

g = {"__name__": "outliers_mod", "spark": object()}
exec(head, g)
analizar_combinacion = g["analizar_combinacion"]
_segmentar = g["_segmentar"]


def make_pdf(valores, dia=1, ventana=24, metrica="ENVIOS"):
    n = len(valores)
    return pd.DataFrame({
        "metrica": [metrica] * n, "canal_dsc": ["HTML"] * n,
        "flujo_dsc": ["Marketing Contextual"] * n, "productos_dsc": ["Venta TC"] * n,
        "objetivo_dsc": ["Ventas"] * n, "dia_semana": [dia] * n, "ventana_meses": [ventana] * n,
        "codmes": [202500 + i + 1 for i in range(n)], "mes_idx": [24000 + i for i in range(n)],
        "valor_mes": valores, "suma_mes": [v * 4 for v in valores], "n_dias": [4] * n,
    })


def main():
    np.random.seed(0)

    # --- Criterio de aceptación: enero=191, resto > 100 000 ---
    valores = [191.0] + list(np.random.uniform(100000, 180000, 23))
    out = analizar_combinacion(make_pdf(valores))
    jan = out[out.valor_mes == 191.0].iloc[0]
    assert bool(jan.es_outlier) and jan.tipo == "BAJO", "FALLO criterio de aceptacion"
    assert jan.metodo == "CLUSTER" and jan.k_elegido == 2
    print("OK  aceptacion: enero=191 -> BAJO por CLUSTER (k=2), frontera_inf=%.1f" % jan.frontera_inf_cluster)

    # --- Pico ALTO aislado (doble cola) ---
    out5 = analizar_combinacion(make_pdf(list(np.random.uniform(1000, 2000, 17)) + [500000.0]))
    hi = out5[out5.valor_mes == 500000.0].iloc[0]
    assert bool(hi.es_outlier) and hi.tipo == "ALTO", "FALLO pico alto"
    print("OK  pico ALTO aislado -> ALTO")

    # --- Continuo unimodal: no debe partir el grupo ---
    out3 = analizar_combinacion(make_pdf([abs(v) for v in np.random.normal(50000, 8000, 18)]))
    assert int(out3.es_outlier.sum()) == 0, "FALLO: unimodal no debe marcar outliers"
    print("OK  unimodal continuo -> 0 outliers")

    # --- Pocos meses -> fallback IQR/MAD ---
    out4 = analizar_combinacion(make_pdf([100.0, 110000, 120000, 115000, 118000]))
    assert out4.metodo.iloc[0] == "IQR_MAD_POCOS_MESES", "FALLO: pocos meses debe ir a fallback"
    print("OK  n=5 -> %s" % out4.metodo.iloc[0])

    # --- Serie constante -> degenerado, sin outliers ---
    out7 = analizar_combinacion(make_pdf([5000.0] * 12))
    assert out7.metodo.iloc[0] == "IQR_MAD_DEGENERADO" and int(out7.es_outlier.sum()) == 0
    print("OK  serie constante -> IQR_MAD_DEGENERADO, 0 outliers")

    # --- Valores <= 0 (guard log) no rompe ---
    out6 = analizar_combinacion(make_pdf([0.0, 0.0] + list(np.random.uniform(100000, 120000, 10))))
    assert out6.metodo.iloc[0] == "CLUSTER"
    print("OK  valores 0 -> guard log OK, ceros BAJO=%s" % out6[out6.valor_mes == 0.0].es_outlier.tolist())

    # --- Método del codo (motor DP) ---
    k, labels, centers, sizes = _segmentar(np.log10(np.array(valores)), 5, "DP", 42, 20)
    assert k == 2
    print("OK  método del codo (DP) reproducible (k=2)")

    # --- Equivalencia DP ≡ SKLEARN sobre casos variados (misma decisión de outliers) ---
    rng = np.random.default_rng(7)
    KEY = ["metodo", "k_elegido", "cluster_rank", "es_outlier", "tipo",
           "frontera_inf_log", "frontera_sup_log", "limite_inf", "limite_sup"]
    mm = 0
    for _ in range(150):
        n = int(rng.integers(6, 25))
        kind = rng.integers(0, 5)
        if kind == 0:
            v = np.abs(rng.normal(50000, 8000, n))
        elif kind == 1:
            v = np.concatenate([[rng.uniform(50, 500)], rng.uniform(80000, 200000, n - 1)])
        elif kind == 2:
            v = np.concatenate([rng.uniform(1000, 3000, n - 1), [rng.uniform(3e5, 9e5)]])
        elif kind == 3:
            v = np.concatenate([rng.uniform(1000, 2000, n // 2), rng.uniform(9e4, 12e4, n - n // 2)])
        else:
            v = 10 ** rng.uniform(2, 5.5, n)
        pdf = make_pdf([float(x) for x in v])
        g["MOTOR_CLUSTER"] = "DP";      od = g["analizar_combinacion"](pdf)
        g["MOTOR_CLUSTER"] = "SKLEARN"; os_ = g["analizar_combinacion"](pdf)
        g["MOTOR_CLUSTER"] = "DP"
        for c in KEY:
            a, b = od[c], os_[c]
            if a.dtype.kind in "fc":
                eq = np.allclose(a.fillna(-999).values, b.fillna(-999).values, rtol=1e-6, atol=1e-9)
            else:
                eq = (a.astype(object).where(a.notna(), None).tolist()
                      == b.astype(object).where(b.notna(), None).tolist())
            if not eq:
                mm += 1
                break
    assert mm == 0, f"DP y SKLEARN difieren en {mm} casos"
    print("OK  equivalencia DP ≡ SKLEARN (150 casos, misma detección)")

    print("\nTODOS LOS CHECKS PASARON")


if __name__ == "__main__":
    main()
