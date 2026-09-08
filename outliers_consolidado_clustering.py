# -*- coding: utf-8 -*-
"""
========================================================================================================
 OUTLIERS MENSUALES CONSOLIDADOS · LÍMITES DINÁMICOS POR CLUSTERING (K-Means 1-D)
 Entorno: Amazon Athena for Apache Spark (PySpark).  Fuente: disc_analyst_intcam.t_ds_agg_comunicaciones
--------------------------------------------------------------------------------------------------------
 EXTENSIÓN de 'outliers_consolidado.py'. Se respeta:
   · el grano  (metrica, canal, flujo, objetivo, producto, dia_semana)  y las ventanas 6/12/18/24,
   · la escala de detección log10 (datos multiplicativos: capta CAÍDAS y PICOS por igual),
   · las columnas configurables y la escritura como TABLA del catálogo Glue.

 QUÉ CAMBIA vs. el script base
 -----------------------------
 El script base fija los cortes en cuartiles (P25/P75) y su cerca de Tukey ±1.5·IQR. Esa cerca es
 RÍGIDA: cuando una combinación tiene dispersión natural amplia en log (p.ej. meses entre 10^3 y 10^5),
 la cerca inferior se abre tanto que un mes genuinamente aislado (191 envíos ≈ 10^2.28) queda DENTRO de
 la cerca y NO se marca. Aquí los cortes se DERIVAN de la estructura de clusters de cada combinación:

   · Se clusteriza (K-Means 1-D) los valores mensuales en log10 de CADA combinación única.
   · K se elige AUTOMÁTICAMENTE por combinación con el método del codo (no se fija k=3).
   · El "grupo principal" = cluster con más meses. Los meses en clusters SEPARADOS por un hueco real:
        - por DEBAJO del principal  -> outlier BAJO  (base de 'limite_inf' = frontera del cluster inferior)
        - por ENCIMA  del principal  -> outlier ALTO   (base de 'limite_sup' = frontera del cluster superior)
   · Se conserva DOBLE COLA y escala log.
   · Se VALIDA la frontera del cluster contra los percentiles de la combinación (P25/P50/P75).
   · IQR/MAD quedan como FALLBACK cuando la muestra es chica (pocos meses), el clustering es
     inestable/degenerado (K=1, error, valores no separables), o sklearn no está disponible.

 TRADE-OFF sklearn (applyInPandas) vs. pyspark.ml.clustering.KMeans  -> ver bloque de diseño abajo y
 el documento DISENO_CLUSTERING.md. Resumen: cada combinación tiene 6–24 puntos; son MILES de modelos
 diminutos. 'pyspark.ml.KMeans' clusteriza UN dataset grande de forma distribuida y habría que lanzarlo
 en bucle por combinación (miles de jobs, cuello de botella en el driver). El patrón correcto para
 "muchos modelos pequeños, uno por grupo" es groupBy(clave).applyInPandas(sklearn): Spark paraleliza
 los grupos y cada worker corre un K-Means local instantáneo sobre <=24 puntos.

 NOTA: aplanado a nivel superior (sin main) para ejecutar por celdas. En Athena Spark 'spark' ya existe.
========================================================================================================
"""

from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType, IntegerType,
                               LongType, DoubleType, BooleanType)

# La sesión 'spark' ya existe en Athena Spark; solo se crea si no estuviera definida.
try:
    spark
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("outliers_clustering_comunicaciones").getOrCreate()

# =========================== CONFIGURACIÓN ===========================
TABLA_FUENTE    = "disc_analyst_intcam.t_ds_agg_comunicaciones"
TABLA_OUTLIERS  = "disc_analyst_intcam.Lista_Outliers_Consolidado_DA"           # outliers consolidados
TABLA_DIAG      = "disc_analyst_intcam.Detalle_Clusters_Outliers_DA"            # detalle mes×combinación (diagnóstico)
S3_OUTLIERS     = "s3://ibk-discovery-ba-us-east-1-992382582498-data/discovery/anl_intcam/BP3616/Outliers/lista_outliers_consolidado"
S3_DIAG         = "s3://ibk-discovery-ba-us-east-1-992382582498-data/discovery/anl_intcam/BP3616/Outliers/detalle_clusters"
ESCRIBIR_DIAG   = True              # además de la lista consolidada, escribir el detalle por combinación

PARAM_CODMES    = "202606"          # mes de evaluación (yyyymm). Las ventanas son los meses PREVIOS.
VENTANAS        = [6, 12, 18, 24]   # ventanas históricas a evaluar
ESCALA_LOG      = True              # detección en escala log10 (recomendado para datos multiplicativos)

# --- Parámetros del método base (fallback IQR/MAD) ---
IQR_FACTOR      = 1.5               # cerca de Tukey (igual que las bandas)
MAD_Z_UMBRAL    = 3.5               # umbral del z-score modificado (criterio robusto)
MIN_MESES       = 4                 # mínimo de meses con dato para permitir flaggear (cualquier método)

# --- Parámetros del CLUSTERING ---
USAR_CLUSTERING = True              # si False, el script se comporta como el base (solo IQR/MAD)
MIN_MESES_CLUSTER = 6               # muestra mínima para intentar clustering; por debajo -> fallback IQR/MAD
K_MAX           = 5                 # tope de K a evaluar en el método del codo
RANDOM_STATE    = 42                # reproducibilidad (igual que el notebook)
N_INIT          = 20               # reinicios de K-Means (igual que el notebook)
GAP_FACTOR      = 1.0               # un cluster se considera AISLADO si el hueco que lo separa del
                                    # principal supera GAP_FACTOR * (dispersión intra-cluster de referencia).
                                    # Evita partir en dos una distribución continua unimodal.
MAX_OUTLIER_FRAC = 0.40            # una "cola" de outliers no puede superar esta fracción de los meses;
                                    # por encima se interpreta como un RÉGIMEN (bimodal), no anomalías.
TOL_PCTL_LOG    = 0.50             # tolerancia (en log10) para validar frontera_cluster ≈ percentil
                                    # (0.50 en log10 ≈ factor 3.16 en unidades originales).
EPS_VALOR       = 1e-9             # guard para log de valores <= 0

# Nombres de columnas de la fuente (ajustar si difieren). metrica -> (columna_fecha, columna_valor)
COL_CANAL       = "canal_dsc"
COL_FLUJO       = "flujo_dsc"
COL_PRODUCTO    = "productos_dsc"
COL_OBJETIVO    = "objetivos_dsc"   # en la fuente suele venir en plural
METRICAS = {
    "ENVIOS": ("fecha_envio_hora_dt", "envios_val"),
    "OPENS":  ("fecha_open_hora_dt",  "opens_val"),
    "CLICKS": ("fecha_click_hora_dt", "clicks_val"),
}
# ====================================================================

# --- Chequeo (informativo) de disponibilidad de sklearn en el DRIVER ---
# Los workers vuelven a importar dentro de la UDF; si no está, hay fallback numpy puro (ver _kmeans_1d).
try:
    import sklearn  # noqa: F401
    print(f"[info] scikit-learn disponible en el driver (v{sklearn.__version__}).")
except Exception as _e:  # pragma: no cover
    print("[warn] scikit-learn NO disponible en el driver. Se usará el K-Means 1-D en numpy (fallback). "
          "Para instalarlo en Athena Spark, añade 'scikit-learn' vía la propiedad de librerías Python "
          "de la sesión/aplicación Spark.")

CELL = ["metrica", "canal_dsc", "flujo_dsc", "productos_dsc", "objetivo_dsc", "dia_semana"]


# ============================================================================================
#  FUNCIONES AUXILIARES (nivel de módulo -> se serializan a los workers con la UDF)
# ============================================================================================
def _kmeans_1d(y, k, random_state=42, n_init=20):
    """K-Means 1-D. Usa sklearn si está; si no, un Lloyd + k-means++ en numpy (determinista).
    Devuelve (labels, centers) con labels en [0..k-1] y centers como array float."""
    import numpy as np
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(y.reshape(-1, 1))
        return labels, km.cluster_centers_.flatten()
    except Exception:
        # ---- Fallback numpy puro (mismo espíritu: n_init reinicios, k-means++) ----
        rng = np.random.RandomState(random_state)
        n = len(y)
        best = None
        for _ in range(n_init):
            # init k-means++ en 1-D
            centers = [y[rng.randint(n)]]
            for _c in range(1, k):
                d2 = np.min(np.stack([(y - c) ** 2 for c in centers], axis=0), axis=0)
                s = d2.sum()
                probs = (d2 / s) if s > 0 else np.full(n, 1.0 / n)
                centers.append(y[rng.choice(n, p=probs)])
            centers = np.array(centers, dtype=float)
            labels = np.zeros(n, dtype=int)
            for _it in range(100):
                labels = np.argmin(np.abs(y.reshape(-1, 1) - centers.reshape(1, -1)), axis=1)
                new = np.array([y[labels == j].mean() if np.any(labels == j) else centers[j]
                                for j in range(k)])
                if np.allclose(new, centers):
                    centers = new
                    break
                centers = new
            inertia = float(np.sum((y - centers[labels]) ** 2))
            if best is None or inertia < best[0]:
                best = (inertia, labels.copy(), centers.copy())
        return best[1], best[2]


def _elegir_k_codo(y, k_max, random_state, n_init):
    """Elige K por el MÉTODO DEL CODO (kneedle: máxima distancia de la curva de inercia a la cuerda
    que une el primer y último punto). Devuelve (k_best, inertias). k_best=1 => sin estructura."""
    import numpy as np
    n = len(y)
    n_unique = int(np.unique(y).size)
    kmax = min(int(k_max), n - 1, n_unique)
    if kmax < 2:
        return 1, [float(np.sum((y - y.mean()) ** 2))]
    inertias = []
    for k in range(1, kmax + 1):
        labels, centers = _kmeans_1d(y, k, random_state, n_init)
        inertias.append(float(np.sum((y - centers[labels]) ** 2)))
    inertias = np.array(inertias, dtype=float)
    ks = np.arange(1, kmax + 1, dtype=float)
    # Normaliza ejes a [0,1] para que la distancia a la cuerda sea comparable
    x = (ks - ks.min()) / (ks.max() - ks.min())
    rng_i = inertias.max() - inertias.min()
    yv = (inertias - inertias.min()) / (rng_i if rng_i > 0 else 1.0)
    x1, y1, x2, y2 = x[0], yv[0], x[-1], yv[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * yv + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    dist = num / den
    k_best = int(ks[int(np.argmax(dist))])          # el "codo"
    return k_best, inertias.tolist()


def analizar_combinacion(pdf):
    """UDF de applyInPandas. Recibe TODOS los meses de UNA combinación (dentro de una ventana) y
    devuelve una fila por mes con: asignación de cluster, fronteras, límites derivados, validación
    contra percentiles, y flags de outlier de doble cola. Todo el cálculo por combinación se hace en
    pandas/numpy porque la muestra es diminuta (6–24 puntos)."""
    import numpy as np
    import pandas as pd

    # --- claves y datos base de la combinación ---
    r0 = pdf.iloc[0]
    ventana = int(r0["ventana_meses"])
    valor = pdf["valor_mes"].astype(float).values
    n = len(pdf)

    # Escala de detección (log10 con guard para valores <= 0)
    if ESCALA_LOG:
        y = np.log10(np.maximum(valor, EPS_VALOR))
    else:
        y = valor.astype(float)

    # --- Estadística robusta base (siempre se calcula: reporte + fallback) ---
    p25_log, p50_log, p75_log = np.percentile(y, [25, 50, 75])
    iqr_log = p75_log - p25_log
    li_iqr_log = p25_log - IQR_FACTOR * iqr_log
    ls_iqr_log = p75_log + IQR_FACTOR * iqr_log
    mad = float(np.median(np.abs(y - p50_log)))
    # z-score modificado robusto; denominador seguro para no dividir por 0 cuando MAD=0 (serie plana).
    mod_z = 0.6745 * (y - p50_log) / mad if mad > 0 else np.zeros_like(y)

    def back(v):
        return float(10.0 ** v) if ESCALA_LOG else float(v)

    # Contenedores de salida (una entrada por mes)
    cluster_id   = np.full(n, -1, dtype=int)
    cluster_rank = np.full(n, -1, dtype=int)
    cluster_cent = np.full(n, np.nan)
    cluster_n    = np.full(n, n, dtype=int)
    es_out       = np.zeros(n, dtype=bool)
    tipo         = np.array([None] * n, dtype=object)

    frontera_inf_log = np.nan
    frontera_sup_log = np.nan
    k_elegido = 0

    # Etiqueta de por qué NO se usó clustering (se sobreescribe a "CLUSTER" si sí se usa).
    if not USAR_CLUSTERING:
        metodo = "IQR_MAD_DESHABILITADO"
    elif n < MIN_MESES_CLUSTER:
        metodo = "IQR_MAD_POCOS_MESES"
    elif np.unique(y).size < 2:
        metodo = "IQR_MAD_DEGENERADO"          # serie constante -> no hay estructura que clusterizar
    else:
        metodo = "IQR_MAD_FALLBACK"

    # --------------------------------------------------------------------------------
    #  RUTA CLUSTERING  (si hay suficientes meses y está habilitado)
    # --------------------------------------------------------------------------------
    usar_cluster = USAR_CLUSTERING and (n >= MIN_MESES_CLUSTER) and (np.unique(y).size >= 2)
    if usar_cluster:
        try:
            k_best, _inertias = _elegir_k_codo(y, K_MAX, RANDOM_STATE, N_INIT)
            if k_best <= 1:
                # Sin estructura de clusters -> el codo no justifica separar -> fallback IQR/MAD.
                metodo = "IQR_MAD_FALLBACK_K1"
            else:
                labels, centers = _kmeans_1d(y, k_best, RANDOM_STATE, N_INIT)
                k_elegido = int(k_best)
                metodo = "CLUSTER"

                # Ordena clusters por centro ascendente -> 'rank' (0 = más bajo)
                orden = np.argsort(centers)
                rank_de = {c: r for r, c in enumerate(orden)}          # label original -> rank
                ranks = np.array([rank_de[l] for l in labels])
                sizes = np.array([int(np.sum(labels == c)) for c in range(len(centers))])

                # Cluster PRINCIPAL = el de mayor tamaño (empate -> el más cercano a la mediana)
                max_size = sizes.max()
                cand = [c for c in range(len(centers)) if sizes[c] == max_size]
                principal = min(cand, key=lambda c: abs(centers[c] - p50_log))
                rank_principal = rank_de[principal]
                minP, maxP = y[labels == principal].min(), y[labels == principal].max()
                spreadP = maxP - minP

                # Reporte por mes: id/rank/centro/tamaño del cluster de cada mes
                cluster_id   = labels.astype(int)
                cluster_rank = ranks.astype(int)
                cluster_cent = np.array([back(centers[l]) for l in labels])
                cluster_n    = np.array([sizes[l] for l in labels], dtype=int)

                # ---- Frontera INFERIOR: cluster inmediatamente por debajo del principal ----
                low_label = orden[rank_principal - 1] if rank_principal - 1 >= 0 else None
                if low_label is not None:
                    L = y[labels == low_label]
                    gap = minP - L.max()
                    ref = max(spreadP, (L.max() - L.min()), 1e-9)
                    frontier = (L.max() + minP) / 2.0
                    below_mask = y <= frontier
                    aislado = (gap >= GAP_FACTOR * ref) and (below_mask.sum() <= MAX_OUTLIER_FRAC * n)
                    if aislado:
                        frontera_inf_log = frontier
                        es_out[below_mask] = True
                        tipo[below_mask] = "BAJO"

                # ---- Frontera SUPERIOR: cluster inmediatamente por encima del principal ----
                high_label = orden[rank_principal + 1] if rank_principal + 1 < len(centers) else None
                if high_label is not None:
                    Hh = y[labels == high_label]
                    gap = Hh.min() - maxP
                    ref = max(spreadP, (Hh.max() - Hh.min()), 1e-9)
                    frontier = (maxP + Hh.min()) / 2.0
                    above_mask = y >= frontier
                    aislado = (gap >= GAP_FACTOR * ref) and (above_mask.sum() <= MAX_OUTLIER_FRAC * n)
                    if aislado:
                        frontera_sup_log = frontier
                        es_out[above_mask] = True
                        tipo[above_mask] = "ALTO"
        except Exception as _err:  # clustering inestable/degenerado -> fallback robusto
            metodo = "IQR_MAD_ERROR"
            usar_cluster = False

    # --------------------------------------------------------------------------------
    #  RUTA FALLBACK  IQR/MAD  (pocos meses, K=1, error, o clustering deshabilitado)
    # --------------------------------------------------------------------------------
    if metodo != "CLUSTER":
        out_iqr = (y < li_iqr_log) | (y > ls_iqr_log)
        out_mad = np.abs(mod_z) > MAD_Z_UMBRAL
        es_out = (n >= MIN_MESES) & (out_iqr | out_mad)
        tipo = np.where(es_out, np.where(y > p50_log, "ALTO", "BAJO"), None).astype(object)

    # --------------------------------------------------------------------------------
    #  LÍMITES REPORTADOS  (frontera de cluster si existe; si no, cerca IQR)
    # --------------------------------------------------------------------------------
    limite_inf_log = frontera_inf_log if not np.isnan(frontera_inf_log) else li_iqr_log
    limite_sup_log = frontera_sup_log if not np.isnan(frontera_sup_log) else ls_iqr_log

    # Validación de la frontera del cluster contra percentiles (en log10)
    dif_inf = (frontera_inf_log - p25_log) if not np.isnan(frontera_inf_log) else np.nan
    dif_sup = (frontera_sup_log - p75_log) if not np.isnan(frontera_sup_log) else np.nan
    valida_inf = (abs(dif_inf) <= TOL_PCTL_LOG) if not np.isnan(dif_inf) else None
    valida_sup = (abs(dif_sup) <= TOL_PCTL_LOG) if not np.isnan(dif_sup) else None

    # --------------------------------------------------------------------------------
    #  Construcción del DataFrame de salida (una fila por mes, orden EXACTO del schema)
    # --------------------------------------------------------------------------------
    out = pd.DataFrame({
        "metrica":        pdf["metrica"].astype(str).values,
        "canal_dsc":      pdf["canal_dsc"].astype(str).values,
        "flujo_dsc":      pdf["flujo_dsc"].astype(str).values,
        "productos_dsc":  pdf["productos_dsc"].astype(str).values,
        "objetivo_dsc":   pdf["objetivo_dsc"].astype(str).values,
        "dia_semana":     pdf["dia_semana"].astype("int32").values,
        "ventana_meses":  np.full(n, ventana, dtype="int32"),
        "codmes":         pdf["codmes"].astype("int32").values,
        "mes_idx":        pdf["mes_idx"].astype("int32").values,
        "valor_mes":      valor.astype("float64"),
        "suma_mes":       pdf["suma_mes"].astype("float64").values,
        "n_dias":         pdf["n_dias"].astype("int64").values,
        "n_meses":        np.full(n, n, dtype="int32"),
        "y_log":          y.astype("float64"),
        "metodo":         np.array([metodo] * n, dtype=object),
        "k_elegido":      np.full(n, k_elegido, dtype="int32"),
        "cluster_id":     cluster_id.astype("int32"),
        "cluster_rank":   cluster_rank.astype("int32"),
        "cluster_center": cluster_cent.astype("float64"),
        "cluster_n":      cluster_n.astype("int32"),
        "q1":             np.full(n, back(p25_log), dtype="float64"),
        "mediana":        np.full(n, back(p50_log), dtype="float64"),
        "q3":             np.full(n, back(p75_log), dtype="float64"),
        "iqr":            np.full(n, back(p75_log) - back(p25_log), dtype="float64"),
        "p25_log":        np.full(n, p25_log, dtype="float64"),
        "p50_log":        np.full(n, p50_log, dtype="float64"),
        "p75_log":        np.full(n, p75_log, dtype="float64"),
        "frontera_inf_cluster": np.full(n, back(frontera_inf_log) if not np.isnan(frontera_inf_log) else np.nan, dtype="float64"),
        "frontera_sup_cluster": np.full(n, back(frontera_sup_log) if not np.isnan(frontera_sup_log) else np.nan, dtype="float64"),
        "frontera_inf_log": np.full(n, frontera_inf_log, dtype="float64"),
        "frontera_sup_log": np.full(n, frontera_sup_log, dtype="float64"),
        "limite_inf":     np.full(n, back(limite_inf_log), dtype="float64"),
        "limite_sup":     np.full(n, back(limite_sup_log), dtype="float64"),
        "limite_inf_iqr": np.full(n, back(li_iqr_log), dtype="float64"),
        "limite_sup_iqr": np.full(n, back(ls_iqr_log), dtype="float64"),
        "dif_inf_p25_log": np.full(n, dif_inf, dtype="float64"),
        "dif_sup_p75_log": np.full(n, dif_sup, dtype="float64"),
        "pctl_valida_inf": np.array([valida_inf] * n, dtype=object),
        "pctl_valida_sup": np.array([valida_sup] * n, dtype=object),
        "mad":            np.full(n, mad, dtype="float64"),
        "mod_zscore":     mod_z.astype("float64"),
        "es_outlier":     es_out.astype(bool),
        "tipo":           tipo,
    })
    return out


# ============================================================================================
#  SCHEMA de salida de applyInPandas (debe coincidir en ORDEN y TIPO con analizar_combinacion)
# ============================================================================================
SCHEMA = StructType([
    StructField("metrica", StringType()),
    StructField("canal_dsc", StringType()),
    StructField("flujo_dsc", StringType()),
    StructField("productos_dsc", StringType()),
    StructField("objetivo_dsc", StringType()),
    StructField("dia_semana", IntegerType()),
    StructField("ventana_meses", IntegerType()),
    StructField("codmes", IntegerType()),
    StructField("mes_idx", IntegerType()),
    StructField("valor_mes", DoubleType()),
    StructField("suma_mes", DoubleType()),
    StructField("n_dias", LongType()),
    StructField("n_meses", IntegerType()),
    StructField("y_log", DoubleType()),
    StructField("metodo", StringType()),
    StructField("k_elegido", IntegerType()),
    StructField("cluster_id", IntegerType()),
    StructField("cluster_rank", IntegerType()),
    StructField("cluster_center", DoubleType()),
    StructField("cluster_n", IntegerType()),
    StructField("q1", DoubleType()),
    StructField("mediana", DoubleType()),
    StructField("q3", DoubleType()),
    StructField("iqr", DoubleType()),
    StructField("p25_log", DoubleType()),
    StructField("p50_log", DoubleType()),
    StructField("p75_log", DoubleType()),
    StructField("frontera_inf_cluster", DoubleType()),
    StructField("frontera_sup_cluster", DoubleType()),
    StructField("frontera_inf_log", DoubleType()),
    StructField("frontera_sup_log", DoubleType()),
    StructField("limite_inf", DoubleType()),
    StructField("limite_sup", DoubleType()),
    StructField("limite_inf_iqr", DoubleType()),
    StructField("limite_sup_iqr", DoubleType()),
    StructField("dif_inf_p25_log", DoubleType()),
    StructField("dif_sup_p75_log", DoubleType()),
    StructField("pctl_valida_inf", BooleanType()),
    StructField("pctl_valida_sup", BooleanType()),
    StructField("mad", DoubleType()),
    StructField("mod_zscore", DoubleType()),
    StructField("es_outlier", BooleanType()),
    StructField("tipo", StringType()),
])


# ============================================================================================
#  1-3) INGESTA / DESPIVOTE / AGREGADO A MES  (idéntico al script base)
# ============================================================================================
eval_idx = int(PARAM_CODMES[:4]) * 12 + int(PARAM_CODMES[4:6])
src = spark.table(TABLA_FUENTE)

partes = []
for _metrica, (_col_fecha, _col_valor) in METRICAS.items():
    partes.append(
        src.where(F.col(_col_fecha).isNotNull() & (F.coalesce(F.col(_col_valor), F.lit(0)) > 0))
           .select(
               F.lit(_metrica).alias("metrica"),
               F.to_date(F.col(_col_fecha)).alias("fecha_dt"),
               F.coalesce(F.when(F.col(COL_CANAL) != "", F.col(COL_CANAL)), F.lit("SIN CANAL")).alias("canal_dsc"),
               F.coalesce(F.when(F.col(COL_FLUJO) != "", F.col(COL_FLUJO)), F.lit("SIN FLUJO")).alias("flujo_dsc"),
               F.coalesce(F.when(F.col(COL_PRODUCTO) != "", F.col(COL_PRODUCTO)), F.lit("SIN PRODUCTO")).alias("productos_dsc"),
               F.coalesce(F.when(F.col(COL_OBJETIVO) != "", F.col(COL_OBJETIVO)), F.lit("SIN OBJETIVO")).alias("objetivo_dsc"),
               F.col(_col_valor).cast("double").alias("cantidad"),
           ))
long_df = partes[0]
for p in partes[1:]:
    long_df = long_df.unionByName(p)

diario = (long_df
    .withColumn("dia_semana", ((F.dayofweek("fecha_dt") + 5) % 7) + 1)   # 1=Lun..7=Dom
    .groupBy("metrica", "canal_dsc", "flujo_dsc", "productos_dsc", "objetivo_dsc", "dia_semana", "fecha_dt")
    .agg(F.sum("cantidad").alias("cantidad_dia")))

mensual = (diario
    .withColumn("codmes", F.date_format("fecha_dt", "yyyyMM").cast("int"))
    .withColumn("mes_idx", F.year("fecha_dt") * 12 + F.month("fecha_dt"))
    .groupBy("metrica", "canal_dsc", "flujo_dsc", "productos_dsc", "objetivo_dsc", "dia_semana", "codmes", "mes_idx")
    .agg(F.avg("cantidad_dia").alias("valor_mes"),
         F.sum("cantidad_dia").alias("suma_mes"),
         F.count(F.lit(1)).alias("n_dias"))
    .withColumn("dif_meses", F.lit(eval_idx) - F.col("mes_idx"))
    .where((F.col("dif_meses") >= 1) & (F.col("dif_meses") <= max(VENTANAS)))
    .cache())


# ============================================================================================
#  4) DETECCIÓN POR VENTANA  ->  clustering por combinación con applyInPandas
# ============================================================================================
def detectar_en_ventana(mensual_df, W):
    df = (mensual_df
          .where(F.col("dif_meses") <= W)
          .withColumn("ventana_meses", F.lit(W).cast("int"))
          .select("metrica", "canal_dsc", "flujo_dsc", "productos_dsc", "objetivo_dsc",
                  "dia_semana", "ventana_meses", "codmes", "mes_idx",
                  "valor_mes", "suma_mes", "n_dias"))
    # Un modelo K-Means 1-D por combinación: Spark paraleliza los grupos, sklearn corre local.
    return df.groupBy(*CELL).applyInPandas(analizar_combinacion, schema=SCHEMA)


resultado = None
for W in VENTANAS:
    r = detectar_en_ventana(mensual, W)
    resultado = r if resultado is None else resultado.unionByName(r)

resultado = resultado.cache()


# ============================================================================================
#  5) LISTA CONSOLIDADA de outliers  (llaves actuales + columnas nuevas solicitadas)
# ============================================================================================
outliers = (resultado
    .where(F.col("es_outlier"))
    .select(
        F.lit(PARAM_CODMES).cast("int").alias("codmes_val"),
        "ventana_meses", "codmes", "metrica", "canal_dsc", "flujo_dsc", "objetivo_dsc", "productos_dsc",
        "dia_semana",
        F.element_at(F.array(*[F.lit(x) for x in ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]]),
                     F.col("dia_semana")).alias("dia_semana_dsc"),
        "tipo", "valor_mes", "suma_mes", "n_dias", "n_meses",
        # --- método y clustering ---
        "metodo", "k_elegido",
        "cluster_id", "cluster_rank", "cluster_center", "cluster_n",
        # --- fronteras de cluster (base de los límites) y límites reportados ---
        "frontera_inf_cluster", "frontera_sup_cluster", "limite_inf", "limite_sup",
        "limite_inf_iqr", "limite_sup_iqr",
        # --- percentiles y validación cluster≈percentil ---
        "q1", "mediana", "q3", "iqr",
        "dif_inf_p25_log", "dif_sup_p75_log", "pctl_valida_inf", "pctl_valida_sup",
        # --- robustos ---
        "mad", "mod_zscore", "es_outlier")
    .orderBy("ventana_meses", "metrica", "canal_dsc", "flujo_dsc", "objetivo_dsc", "productos_dsc",
             "dia_semana", "codmes"))


# ============================================================================================
#  6) ESCRITURA como TABLA del catálogo Glue (particionada por ventana)
# ============================================================================================
spark.sql(f"DROP TABLE IF EXISTS {TABLA_OUTLIERS}")
(outliers.write.mode("overwrite").format("parquet")
    .option("path", S3_OUTLIERS)
    .partitionBy("ventana_meses")
    .saveAsTable(TABLA_OUTLIERS))

if ESCRIBIR_DIAG:
    # Detalle mes×combinación (todas las filas, sean outlier o no): permite auditar clusters,
    # fronteras y la validación frontera≈percentil por combinación.
    diag = resultado.select(
        F.lit(PARAM_CODMES).cast("int").alias("codmes_val"),
        "ventana_meses", "codmes", "metrica", "canal_dsc", "flujo_dsc", "objetivo_dsc", "productos_dsc",
        "dia_semana", "valor_mes", "suma_mes", "n_dias", "n_meses", "y_log",
        "metodo", "k_elegido", "cluster_id", "cluster_rank", "cluster_center", "cluster_n",
        "q1", "mediana", "q3", "iqr", "p25_log", "p50_log", "p75_log",
        "frontera_inf_cluster", "frontera_sup_cluster", "frontera_inf_log", "frontera_sup_log",
        "limite_inf", "limite_sup", "limite_inf_iqr", "limite_sup_iqr",
        "dif_inf_p25_log", "dif_sup_p75_log", "pctl_valida_inf", "pctl_valida_sup",
        "mad", "mod_zscore", "es_outlier", "tipo")
    spark.sql(f"DROP TABLE IF EXISTS {TABLA_DIAG}")
    (diag.write.mode("overwrite").format("parquet")
        .option("path", S3_DIAG)
        .partitionBy("ventana_meses")
        .saveAsTable(TABLA_DIAG))


# ============================================================================================
#  7) RESUMEN + PRUEBA DEL CRITERIO DE ACEPTACIÓN
# ============================================================================================
print("=== Outliers por ventana y método ===")
(resultado.where(F.col("es_outlier"))
    .groupBy("ventana_meses", "metodo")
    .agg(F.count(F.lit(1)).alias("n_outliers"),
         F.countDistinct("metrica", "canal_dsc", "flujo_dsc", "objetivo_dsc", "productos_dsc", "dia_semana")
          .alias("combinaciones_afectadas"))
    .orderBy("ventana_meses", "metodo").show(truncate=False))

print("=== Criterio de aceptación: Venta TC / Marketing Contextual / Ventas / HTML / ENVIOS / Lunes / Enero ===")
(resultado
    .where((F.col("metrica") == "ENVIOS") & (F.col("dia_semana") == 1)
           & (F.upper(F.col("productos_dsc")).contains("VENTA TC"))
           & (F.upper(F.col("flujo_dsc")).contains("MARKETING CONTEXTUAL"))
           & (F.upper(F.col("objetivo_dsc")).contains("VENTAS"))
           & (F.upper(F.col("canal_dsc")).contains("HTML"))
           & (F.col("codmes") % 100 == 1))   # enero
    .select("ventana_meses", "codmes", "valor_mes", "metodo", "k_elegido",
            "cluster_rank", "cluster_n", "frontera_inf_cluster", "limite_inf",
            "limite_inf_iqr", "es_outlier", "tipo")
    .orderBy("ventana_meses").show(truncate=False))

print(f"Tabla de outliers creada: {TABLA_OUTLIERS}")
if ESCRIBIR_DIAG:
    print(f"Tabla de detalle/diagnóstico creada: {TABLA_DIAG}")
