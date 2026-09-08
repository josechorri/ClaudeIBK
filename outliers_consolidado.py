# -*- coding: utf-8 -*-
"""
========================================================================================================
 OUTLIERS MENSUALES CONSOLIDADOS · MÚLTIPLES VENTANAS (6/12/18/24 meses)
 Entorno: Amazon Athena for Apache Spark (PySpark).  Fuente: disc_analyst_intcam.t_ds_agg_comunicaciones
--------------------------------------------------------------------------------------------------------
 Salida: una TABLA consolidada con TODOS los outliers identificados, al nivel de todas las combinaciones
         de (ventana, mes, flujo, objetivo, producto, metrica).
 Método: por cada ventana y cada combinación (flujo, objetivo, producto, metrica) se toma un valor por
         mes (promedio diario del mes) y se marca outlier el mes que quede fuera de la cerca Tukey IQR
         (±1.5) o supere el z-score modificado por MAD (>3.5). Guarda de mínimo de meses para no flaggear
         sobre muestras diminutas.
 Uso posterior: al construir las bandas de la ventana W, LEFT ANTI JOIN contra esta tabla por
         (ventana_meses, metrica, flujo, objetivo, producto, codmes) para descartar los meses atípicos.
 NOTA: el script está aplanado a nivel superior (sin función main) para evitar errores de indentación al
       ejecutarlo por celdas. En Athena Spark la sesión 'spark' ya existe.
========================================================================================================
"""

from pyspark.sql import functions as F

# La sesión 'spark' ya existe en Athena Spark; solo se crea si no estuviera definida.
try:
    spark
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("outliers_consolidado_comunicaciones").getOrCreate()

# =========================== CONFIGURACIÓN ===========================
TABLA_FUENTE    = "disc_analyst_intcam.t_ds_agg_comunicaciones"
TABLA_OUTLIERS  = "disc_analyst_intcam.Lista_Outliers_Consolidado_DA"   # tabla de salida (catálogo Glue)
S3_OUTLIERS     = "s3://ibk-discovery-ba-us-east-1-992382582498-data/discovery/anl_intcam/BP3616/Outliers/lista_outliers_consolidado"

PARAM_CODMES    = "202606"          # mes de evaluación (yyyymm). Las ventanas son los meses PREVIOS.
VENTANAS        = [6, 12, 18, 24]   # ventanas históricas a evaluar
IQR_FACTOR      = 1.5               # cerca de Tukey (igual que las bandas)
ESCALA_LOG      = True              # detectar en escala log10: capta CAÍDAS y PICOS por igual
                                    # (datos multiplicativos). Si False, usa escala lineal.
MAD_Z_UMBRAL    = 3.5               # umbral del z-score modificado (criterio robusto)
MIN_MESES       = 4                 # mínimo de meses con dato para permitir flaggear

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

eval_idx = int(PARAM_CODMES[:4]) * 12 + int(PARAM_CODMES[4:6])
src = spark.table(TABLA_FUENTE)

# ---------- 1) Despivote a filas (metrica, fecha_dt, flujo, producto, objetivo, cantidad) ----------
partes = []
for _metrica, (_col_fecha, _col_valor) in METRICAS.items():
    partes.append(
        src.where(F.col(_col_fecha).isNotNull() & (F.coalesce(F.col(_col_valor), F.lit(0)) > 0))
           .select(
               F.lit(_metrica).alias("metrica"),
               F.to_date(F.col(_col_fecha)).alias("fecha_dt"),                 # asume hora efectiva (Lima)
               F.coalesce(F.when(F.col(COL_CANAL) != "", F.col(COL_CANAL)), F.lit("SIN CANAL")).alias("canal_dsc"),  # CAMBIO: + canal
               F.coalesce(F.when(F.col(COL_FLUJO)    != "", F.col(COL_FLUJO)),    F.lit("SIN FLUJO")).alias("flujo_dsc"),
               F.coalesce(F.when(F.col(COL_PRODUCTO) != "", F.col(COL_PRODUCTO)), F.lit("SIN PRODUCTO")).alias("productos_dsc"),
               F.coalesce(F.when(F.col(COL_OBJETIVO) != "", F.col(COL_OBJETIVO)), F.lit("SIN OBJETIVO")).alias("objetivo_dsc"),
               F.col(_col_valor).cast("double").alias("cantidad"),
           ))
long_df = partes[0]
for p in partes[1:]:
    long_df = long_df.unionByName(p)

# ---------- 2) Agregado a nivel DÍA por combinación ----------
diario = (long_df
    .withColumn("dia_semana", ((F.dayofweek("fecha_dt") + 5) % 7) + 1)   # CAMBIO: 1=Lun..7=Dom (dayofweek de Spark arranca en Dom)
    .groupBy("metrica", "canal_dsc", "flujo_dsc", "productos_dsc", "objetivo_dsc", "dia_semana", "fecha_dt")   # CAMBIO: + canal
    .agg(F.sum("cantidad").alias("cantidad_dia")))

# ---------- 3) Valor por MES y combinación (promedio diario del mes) + distancia en meses ----------
mensual = (diario
    .withColumn("codmes", F.date_format("fecha_dt", "yyyyMM").cast("int"))
    .withColumn("mes_idx", F.year("fecha_dt") * 12 + F.month("fecha_dt"))
    .groupBy("metrica", "canal_dsc", "flujo_dsc", "productos_dsc", "objetivo_dsc", "dia_semana", "codmes", "mes_idx")   # CAMBIO: + canal, + dia_semana
    .agg(F.avg("cantidad_dia").alias("valor_mes"),
         F.sum("cantidad_dia").alias("suma_mes"),
         F.count(F.lit(1)).alias("n_dias"))
    .withColumn("dif_meses", F.lit(eval_idx) - F.col("mes_idx"))
    .where((F.col("dif_meses") >= 1) & (F.col("dif_meses") <= max(VENTANAS)))
    .cache())

# ---------- 4) Detección de outliers por ventana ----------
CELL = ["metrica", "canal_dsc", "flujo_dsc", "productos_dsc", "objetivo_dsc", "dia_semana"]   # CAMBIO: + canal, + dia_semana

def detectar_en_ventana(mensual_df, W):
    df = mensual_df.where(F.col("dif_meses") <= W)

    # Escala de DETECCIÓN. En log10 la cerca inferior nunca se vuelve negativa, así que las
    # caídas (p.ej. 191 vs. 6 cifras) se detectan igual que los picos. Guard contra valores <= 0.
    if ESCALA_LOG:
        df = df.withColumn("y", F.log10(F.greatest(F.col("valor_mes"), F.lit(1e-9))))
    else:
        df = df.withColumn("y", F.col("valor_mes"))

    # Cuartiles y cercas de Tukey sobre la escala de detección
    stats = (df.groupBy(*CELL).agg(
        F.percentile_approx(F.col("y"), F.array(F.lit(0.25), F.lit(0.5), F.lit(0.75)), F.lit(10000)).alias("pctls"),
        F.count(F.lit(1)).alias("n_meses")))
    stats = (stats
        .withColumn("q1_y", F.col("pctls")[0])
        .withColumn("med_y", F.col("pctls")[1])
        .withColumn("q3_y", F.col("pctls")[2])
        .withColumn("li_y", F.col("q1_y") - F.lit(IQR_FACTOR) * (F.col("q3_y") - F.col("q1_y")))
        .withColumn("ls_y", F.col("q3_y") + F.lit(IQR_FACTOR) * (F.col("q3_y") - F.col("q1_y")))
        .drop("pctls"))
    df = df.join(stats, CELL, "inner")

    # MAD sobre la escala de detección
    df = df.withColumn("abs_dev", F.abs(F.col("y") - F.col("med_y")))
    mad = df.groupBy(*CELL).agg(F.percentile_approx(F.col("abs_dev"), F.lit(0.5), F.lit(10000)).alias("mad"))
    df = df.join(mad, CELL, "inner")

    # Flags de outlier de DOS COLAS (alto y bajo) sobre la escala de detección
    df = (df
        .withColumn("mod_zscore",
            F.when(F.col("mad") > 0, F.lit(0.6745) * (F.col("y") - F.col("med_y")) / F.col("mad")).otherwise(F.lit(0.0)))
        .withColumn("out_iqr", (F.col("y") < F.col("li_y")) | (F.col("y") > F.col("ls_y")))
        .withColumn("out_mad", F.abs(F.col("mod_zscore")) > F.lit(MAD_Z_UMBRAL))
        .withColumn("es_outlier", (F.col("n_meses") >= F.lit(MIN_MESES)) & (F.col("out_iqr") | F.col("out_mad")))
        .withColumn("tipo", F.when(~F.col("es_outlier"), F.lit(None))
                             .when(F.col("y") > F.col("med_y"), F.lit("ALTO")).otherwise(F.lit("BAJO")))
        .withColumn("ventana_meses", F.lit(W)))

    # Umbrales y cuartiles REPORTADOS en unidades originales (se des-transforman si hubo log).
    # Nota: 'mad' y 'mod_zscore' quedan en la escala de detección (log si ESCALA_LOG=True).
    back = (lambda c: F.pow(F.lit(10.0), F.col(c))) if ESCALA_LOG else (lambda c: F.col(c))
    df = (df
        .withColumn("q1", back("q1_y"))
        .withColumn("mediana", back("med_y"))
        .withColumn("q3", back("q3_y"))
        .withColumn("iqr", F.col("q3") - F.col("q1"))
        .withColumn("limite_inf", back("li_y"))
        .withColumn("limite_sup", back("ls_y")))
    return df

resultado = None
for W in VENTANAS:
    r = detectar_en_ventana(mensual, W)
    resultado = r if resultado is None else resultado.unionByName(r)

# ---------- 5) Lista CONSOLIDADA de outliers (todas las ventanas) ----------
outliers = (resultado
    .where(F.col("es_outlier"))
    .select(
        F.lit(PARAM_CODMES).cast("int").alias("codmes_val"),                               # CAMBIO: mes consultado
        "ventana_meses", "codmes", "metrica", "canal_dsc", "flujo_dsc", "objetivo_dsc", "productos_dsc",
        "dia_semana",                                                                      # CAMBIO: + dia_semana
        F.element_at(F.array(*[F.lit(x) for x in ["Lun","Mar","Mie","Jue","Vie","Sab","Dom"]]),
                     F.col("dia_semana")).alias("dia_semana_dsc"),                          # CAMBIO: etiqueta Lun..Dom
        "tipo", "valor_mes", "suma_mes", "n_dias", "n_meses",
        "q1", "mediana", "q3", "iqr", "limite_inf", "limite_sup", "mad", "mod_zscore")
    .orderBy("ventana_meses", "metrica", "canal_dsc", "flujo_dsc", "objetivo_dsc", "productos_dsc", "dia_semana", "codmes"))   # CAMBIO: + canal, + dia_semana

# ---------- 6) Escritura como TABLA del catálogo Glue (particionada por ventana) ----------
spark.sql(f"DROP TABLE IF EXISTS {TABLA_OUTLIERS}")
(outliers.write.mode("overwrite").format("parquet")
    .option("path", S3_OUTLIERS)
    .partitionBy("ventana_meses")
    .saveAsTable(TABLA_OUTLIERS))

# ---------- 7) Resumen en log ----------
print("=== Outliers por ventana ===")
(resultado.where(F.col("es_outlier"))
    .groupBy("ventana_meses")
    .agg(F.count(F.lit(1)).alias("n_outliers"),
         F.countDistinct("metrica", "canal_dsc", "flujo_dsc", "objetivo_dsc", "productos_dsc", "dia_semana").alias("combinaciones_afectadas"))
    .orderBy("ventana_meses").show(truncate=False))

print(f"Tabla creada: {TABLA_OUTLIERS}")
