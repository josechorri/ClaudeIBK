# -*- coding: utf-8 -*-
"""
========================================================================================================
 OUTLIERS MENSUALES CONSOLIDADOS · LÍMITES DINÁMICOS POR CLUSTERING (K-Means 1-D)
 Entorno: Amazon Athena for Apache Spark (PySpark).  Fuente: disc_analyst_intcam.t_ds_agg_comunicaciones
--------------------------------------------------------------------------------------------------------
 EXTENSIÓN de 'outliers_consolidado.py'. Se respeta:
   · el grano  (metrica, canal, flujo, objetivo, producto, dia_semana)  y la ventana estándar de 6 meses,
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
     inestable/degenerado (K=1, error, valores no separables).

 ============================== OPTIMIZACIÓN DE RENDIMIENTO ==============================
 La versión anterior corría K-Means iterativo de sklearn DENTRO de applyInPandas y, por combinación,
 hacía el método del codo (k=1..5) con n_init=20 reinicios cada uno -> ~100 ajustes iterativos + el
 ajuste final, multiplicado por decenas de miles de combinaciones. Ese era el 40-min de cómputo.

 Como el problema es 1-D con muy pocos puntos (6–24), el óptimo GLOBAL de K-Means en 1-D se calcula de
 forma EXACTA y determinista por PROGRAMACIÓN DINÁMICA (algoritmo tipo Ckmeans.1d.dp) en O(K·n²):
   · una sola pasada de DP entrega la INERCIA óptima de TODOS los K a la vez (el codo sale gratis) y la
     segmentación óptima para el K elegido;
   · es exactamente el óptimo que 'n_init=20' de sklearn intentaba aproximar -> MISMO resultado, pero
     ~100× menos operaciones y sin dependencia de scikit-learn en los workers;
   · determinista => reproducibilidad total (no depende de random_state).
 El motor sklearn se conserva disponible (MOTOR_CLUSTER="SKLEARN") para auditoría/reproducción, pero el
 DP es el motor por defecto. La equivalencia de salida (es_outlier, tipo, k_elegido, fronteras, límites)
 está verificada en tests/test_clustering_logic.py.

 Otras optimizaciones (no alteran el output, solo la velocidad):
   · Se evita recomputar el ajuste final (la propia DP/elbow ya deja la segmentación del K elegido).
   · Arrow habilitado para applyInPandas; se proyectan SOLO las columnas necesarias antes del groupBy.
   · 'resultado' se materializa una vez (cache) porque lo consumen 4 acciones (2 escrituras + 2 resúmenes).

 TRADE-OFF sklearn vs. pyspark.ml.KMeans -> ver DISENO_CLUSTERING.md. En síntesis: son MILES de modelos
 diminutos (uno por combinación), así que el patrón correcto es groupBy(clave).applyInPandas(...) con un
 solver 1-D local; 'pyspark.ml.KMeans' clusteriza UN dataset grande y habría que lanzarlo en bucle por
 combinación (miles de jobs, cuello de botella en el driver). Dentro de la UDF, el solver 1-D exacto (DP)
 es superior al K-Means iterativo para este grano.

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

# Arrow acelera el transporte de datos hacia/desde la UDF de pandas (no cambia resultados).
try:
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
except Exception:
    pass

# =========================== CONFIGURACIÓN ===========================
TABLA_FUENTE    = "disc_analyst_intcam.t_ds_agg_comunicaciones"
TABLA_OUTLIERS  = "disc_analyst_intcam.Lista_Outliers_Consolidado_DA"           # outliers consolidados
TABLA_DIAG      = "disc_analyst_intcam.Detalle_Clusters_Outliers_DA"            # detalle mes×combinación (diagnóstico)
S3_OUTLIERS     = "s3://ibk-discovery-ba-us-east-1-992382582498-data/discovery/anl_intcam/BP3616/Outliers/lista_outliers_consolidado"
S3_DIAG         = "s3://ibk-discovery-ba-us-east-1-992382582498-data/discovery/anl_intcam/BP3616/Outliers/detalle_clusters"
ESCRIBIR_DIAG   = True              # además de la lista consolidada, escribir el detalle por combinación

PARAM_CODMES    = "202606"          # mes de evaluación (yyyymm). Las ventanas son los meses PREVIOS.
VENTANAS        = [6]               # ventana ESTÁNDAR: una sola ventana de 6 meses
ESCALA_LOG      = True              # detección en escala log10 (recomendado para datos multiplicativos)

# --- Parámetros del método base (fallback IQR/MAD) ---
IQR_FACTOR      = 1.5               # cerca de Tukey (igual que las bandas)
MAD_Z_UMBRAL    = 3.5               # umbral del z-score modificado (criterio robusto)
MIN_MESES       = 4                 # mínimo de meses con dato para permitir flaggear (cualquier método)

# --- Parámetros del CLUSTERING ---
USAR_CLUSTERING = True              # si False, el script se comporta como el base (solo IQR/MAD)
MIN_MESES_CLUSTER = 6               # muestra mínima para intentar clustering; por debajo -> fallback IQR/MAD
K_MAX           = 5                 # tope de K a evaluar en el método del codo
MOTOR_CLUSTER   = "DP"              # "DP" = k-means 1-D EXACTO por prog. dinámica (rápido, determinista).
                                    # "SKLEARN" = K-Means iterativo (auditoría/compatibilidad).
RANDOM_STATE    = 42                # reproducibilidad (solo aplica al motor SKLEARN; el DP es determinista)
N_INIT          = 20               # reinicios de K-Means (solo aplica al motor SKLEARN)
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

CELL = ["metrica", "canal_dsc", "flujo_dsc", "productos_dsc", "objetivo_dsc", "dia_semana"]


# ============================================================================================
#  FUNCIONES AUXILIARES (nivel de módulo -> se serializan a los workers con la UDF)
# ============================================================================================
def _kneedle(inertias):
    """Método del codo: K = punto de la curva de inercia a máxima distancia de la cuerda que une el
    primer y último punto (ejes normalizados a [0,1]). Devuelve K en [1..len(inertias)]."""
    import numpy as np
    inertias = np.asarray(inertias, dtype=float)
    m = len(inertias)
    if m < 2:
        return 1
    ks = np.arange(1, m + 1, dtype=float)
    x = (ks - ks.min()) / (ks.max() - ks.min())
    rng_i = inertias.max() - inertias.min()
    yv = (inertias - inertias.min()) / (rng_i if rng_i > 0 else 1.0)
    x1, y1, x2, y2 = x[0], yv[0], x[-1], yv[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * yv + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    return int(ks[int(np.argmax(num / den))])


def _segmentar_dp(y, k_max):
    """K-Means 1-D EXACTO por programación dinámica (tipo Ckmeans.1d.dp).
    Devuelve (inertias, seg_for) donde:
      · inertias[k-1] = inercia (SSE) ÓPTIMA con k clusters, para k=1..kmax;
      · seg_for(k) -> (labels, centers, sizes) con labels ya en RANGO (0=cluster más bajo),
        centers ordenados ascendentemente y sizes por rango.
    Los clusters óptimos en 1-D son intervalos contiguos sobre los valores ordenados; la DP los halla
    en O(k·n²) usando sumas prefijas para el costo de cada segmento en O(1)."""
    import numpy as np
    order = np.argsort(y, kind="mergesort")     # estable
    s = y[order]
    n = len(s)
    kmax = min(int(k_max), n)
    # Sumas prefijas para SSE de un segmento sorted[a..b] (inclusive) en O(1).
    P = np.concatenate(([0.0], np.cumsum(s)))
    Q = np.concatenate(([0.0], np.cumsum(s * s)))

    def cost(a, b):
        cnt = b - a + 1
        sm = P[b + 1] - P[a]
        v = (Q[b + 1] - Q[a]) - sm * sm / cnt
        return v if v > 0.0 else 0.0

    INF = float("inf")
    # D[k][i] = SSE mínima al agrupar los primeros i puntos ordenados en k clusters.
    D = [[INF] * (n + 1) for _ in range(kmax + 1)]
    B = [[0] * (n + 1) for _ in range(kmax + 1)]   # backtracking del corte
    D[0][0] = 0.0
    for k in range(1, kmax + 1):
        for i in range(k, n + 1):
            best, bestj = INF, k - 1
            for j in range(k - 1, i):              # j = nº de puntos en los primeros k-1 clusters
                dj = D[k - 1][j]
                if dj == INF:
                    continue
                c = dj + cost(j, i - 1)
                if c < best:
                    best, bestj = c, j
            D[k][i], B[k][i] = best, bestj
    inertias = [D[k][n] for k in range(1, kmax + 1)]

    def seg_for(k):
        bounds = []
        i, kk = n, k
        while kk > 0:
            j = B[kk][i]
            bounds.append((j, i - 1))
            i, kk = j, kk - 1
        bounds.reverse()                            # ascendente (rango 0 = más bajo)
        labels_sorted = np.empty(n, dtype=int)
        centers = np.empty(k, dtype=float)
        sizes = np.zeros(k, dtype=int)
        for rank, (a, b) in enumerate(bounds):
            labels_sorted[a:b + 1] = rank
            centers[rank] = s[a:b + 1].mean()
            sizes[rank] = b - a + 1
        labels = np.empty(n, dtype=int)
        labels[order] = labels_sorted               # de vuelta al orden original
        return labels, centers, sizes

    return inertias, seg_for


def _kmeans_1d_sklearn(y, k, random_state=42, n_init=20):
    """K-Means 1-D iterativo (motor de compatibilidad). Usa sklearn si está; si no, Lloyd+kmeans++ numpy.
    Devuelve (labels, centers)."""
    import numpy as np
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(y.reshape(-1, 1))
        return labels, km.cluster_centers_.flatten()
    except Exception:
        rng = np.random.RandomState(random_state)
        n = len(y)
        best = None
        for _ in range(n_init):
            centers = [y[rng.randint(n)]]
            for _c in range(1, k):
                d2 = np.min(np.stack([(y - c) ** 2 for c in centers], axis=0), axis=0)
                sdt = d2.sum()
                probs = (d2 / sdt) if sdt > 0 else np.full(n, 1.0 / n)
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


def _segmentar(y, k_max, motor, random_state, n_init):
    """Unifica ambos motores. Devuelve (k_best, labels, centers, sizes) con labels EN RANGO
    (0 = cluster más bajo) y centers ordenados ascendentemente. k_best=1 => sin estructura (fallback)."""
    import numpy as np
    n = len(y)
    kmax = min(int(k_max), n - 1, int(np.unique(y).size))
    if kmax < 2:
        return 1, None, None, None

    if motor == "DP":
        inertias, seg_for = _segmentar_dp(y, kmax)
        k_best = _kneedle(inertias)
        if k_best <= 1:
            return 1, None, None, None
        labels, centers, sizes = seg_for(k_best)
        return k_best, labels, centers, sizes

    # -------- motor SKLEARN (compatibilidad) --------
    inertias = []
    for k in range(1, kmax + 1):
        lb, ct = _kmeans_1d_sklearn(y, k, random_state, n_init)
        inertias.append(float(np.sum((y - ct[lb]) ** 2)))
    k_best = _kneedle(inertias)
    if k_best <= 1:
        return 1, None, None, None
    labels, centers = _kmeans_1d_sklearn(y, k_best, random_state, n_init)
    # Reordena etiquetas a RANGO por centro ascendente (para unificar con el DP).
    orden = np.argsort(centers)
    rank_de = {int(c): r for r, c in enumerate(orden)}
    labels_rank = np.array([rank_de[int(l)] for l in labels], dtype=int)
    centers_sorted = centers[orden]
    sizes = np.array([int(np.sum(labels_rank == r)) for r in range(k_best)], dtype=int)
    return k_best, labels_rank, centers_sorted, sizes


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
            k_best, labels, centers, sizes = _segmentar(y, K_MAX, MOTOR_CLUSTER, RANDOM_STATE, N_INIT)
            if k_best <= 1:
                # Sin estructura de clusters -> el codo no justifica separar -> fallback IQR/MAD.
                metodo = "IQR_MAD_FALLBACK_K1"
            else:
                k_elegido = int(k_best)
                metodo = "CLUSTER"

                # labels ya vienen EN RANGO (0 = cluster más bajo); centers ordenados ascendentemente.
                # Cluster PRINCIPAL = el de mayor tamaño (empate -> el más cercano a la mediana).
                max_size = int(sizes.max())
                cand = [r for r in range(k_best) if sizes[r] == max_size]
                principal = min(cand, key=lambda r: abs(centers[r] - p50_log))
                minP, maxP = y[labels == principal].min(), y[labels == principal].max()
                spreadP = maxP - minP

                # Reporte por mes: id/rank/centro/tamaño del cluster de cada mes.
                cluster_id   = labels.astype(int)
                cluster_rank = labels.astype(int)
                cluster_cent = np.array([back(centers[l]) for l in labels])
                cluster_n    = sizes[labels].astype(int)

                # ---- Frontera INFERIOR: cluster inmediatamente por debajo del principal ----
                if principal - 1 >= 0:
                    L = y[labels == principal - 1]
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
                if principal + 1 < k_best:
                    Hh = y[labels == principal + 1]
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
    # Un solver 1-D por combinación: Spark paraleliza los grupos, el DP corre local (microsegundos).
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
