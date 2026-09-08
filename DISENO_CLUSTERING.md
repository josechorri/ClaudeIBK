# Detección de outliers con límites dinámicos por clustering (K-Means 1-D)

Extensión de `outliers_consolidado.py` → `outliers_consolidado_clustering.py`.
Documento de diseño de los tres entregables.

---

## 1. Problema con los límites fijos P25/P75

El script base marca outlier el mes que cae fuera de la **cerca de Tukey** `P25 − 1.5·IQR` /
`P75 + 1.5·IQR`, calculada en `log10` por combinación
`(metrica, canal, flujo, objetivo, producto, dia_semana)` y ventana (6/12/18/24 meses).

Esa cerca es **rígida**: su ancho solo depende del IQR. Cuando una combinación tiene dispersión
natural amplia (meses legítimos entre `10³` y `10⁵`), la cerca inferior se abre tanto que un mes
genuinamente anómalo (p.ej. 191 envíos ≈ `10^2.28`) puede quedar **dentro** y no marcarse; y al revés,
en combinaciones muy planas, cualquier variación menor se marca. No se adapta a la **forma** real de
la distribución de cada combinación.

La idea del notebook (`statistics.ipynb`) es hallar los **grupos naturales** de cada serie con
K-Means 1-D y usar sus fronteras como cortes, validándolos contra P25/P50/P75. Aquí se lleva ese
enfoque, por combinación, al pipeline PySpark.

---

## 2. Cómo se derivan `limite_inf` / `limite_sup` de los clusters

Por cada combinación **única** y ventana:

1. **Escala.** Se trabaja en `y = log10(max(valor_mes, 1e-9))` (multiplicativo; doble cola: capta
   caídas y picos por igual; guard contra valores `≤ 0`).
2. **K automático (método del codo).** Se evalúa la inercia de K-Means para `k = 1..Kmax`
   (`Kmax = min(K_MAX=5, n−1, nº valores distintos)`) y se elige el **codo** con el criterio *kneedle*:
   el K cuya inercia queda a máxima distancia de la cuerda que une el primer y último punto de la
   curva. **No se fija k=3.**
3. **Grupo principal.** Se ordenan los clusters por su centro y se toma como *principal* el de **mayor
   número de meses** (empate → el más cercano a la mediana). Es el "comportamiento normal".
4. **Fronteras.**
   - Cluster inmediatamente **por debajo** del principal → si está **aislado**, su frontera con el
     principal es la base de `limite_inf`:
     `frontera_inf = (max(cluster_inferior) + min(principal)) / 2` (en `log10`).
   - Cluster inmediatamente **por encima** → análogo para `limite_sup`.
   - Un mes por debajo de `frontera_inf` se marca **BAJO**; por encima de `frontera_sup`, **ALTO**.
5. **Criterio de "aislado" (evita partir en dos una distribución continua).** El hueco que separa el
   cluster candidato del principal debe superar la dispersión intra-cluster de referencia:
   `gap ≥ GAP_FACTOR · max(spread_principal, spread_candidato)` **y** la cola no puede exceder
   `MAX_OUTLIER_FRAC` (40 %) de los meses (por encima se interpreta como **régimen bimodal**, no
   anomalía). Si no está aislado, ese lado usa la cerca IQR como `limite`.

`limite_inf`/`limite_sup` reportados = **frontera de cluster** cuando existe una cola aislada; en caso
contrario, la **cerca IQR** de ese lado (`limite_inf_iqr`/`limite_sup_iqr` se reportan siempre para
comparar). Todo se des-transforma a unidades originales (`10^y`) en la salida; `mad`/`mod_zscore`
quedan en la escala de detección.

### Por qué esto captura el caso de aceptación
`Venta TC / Marketing Contextual / Ventas / HTML / ENVIOS / Lunes`, con enero ≈ 191 y el resto de meses
> 100 000: el codo elige **K=2**, el mes de enero queda **solo** en el cluster inferior (`cluster_n=1`),
el hueco `log10(≈130000) − log10(191) ≈ 2.8` supera con creces la dispersión del grupo principal, y la
cola es 1/24 < 40 %. → enero se marca **BAJO** y `frontera_inf_cluster` se sitúa entre 191 y 100 000.
Verificado en pruebas (`tests/test_clustering_logic.py`).

---

## 3. Validación contra percentiles y fallback

- **Validación.** Se calculan `p25/p50/p75` (log) por combinación y se compara la frontera del cluster
  con el percentil correspondiente: `dif_inf_p25_log = frontera_inf_log − p25_log`,
  `pctl_valida_inf = |dif| ≤ TOL_PCTL_LOG` (0.50 en log ≈ factor 3.16). Esto reproduce la comparación
  "centros/fronteras vs percentiles" del notebook y deja auditar cuándo el clustering coincide con los
  cuartiles y cuándo se separa (que es justo cuando aporta).
- **IQR/MAD como fallback**, cuando el clustering no es fiable:
  - `n < MIN_MESES_CLUSTER` (6 meses) → `IQR_MAD_POCOS_MESES`.
  - Serie constante / < 2 valores distintos → `IQR_MAD_DEGENERADO`.
  - El codo devuelve `K=1` (sin estructura) → `IQR_MAD_FALLBACK_K1`.
  - Error/inestabilidad de K-Means → `IQR_MAD_ERROR`.
  - `USAR_CLUSTERING=False` → comportamiento idéntico al script base.
  En fallback se aplica exactamente el criterio original (`out_iqr | out_mad`, con `MIN_MESES=4`).
- **Reproducibilidad.** `random_state=42`, `n_init=20` (igual que el notebook), tanto en sklearn como
  en el fallback numpy.

---

## 4. Trade-off: `applyInPandas` + sklearn  vs.  `pyspark.ml.clustering.KMeans`

**Decisión: `groupBy(clave).applyInPandas(sklearn KMeans)`.**

| Criterio | `applyInPandas` + sklearn (elegido) | `pyspark.ml` KMeans distribuido |
|---|---|---|
| Modelo de cómputo | **Muchos modelos pequeños**, uno por combinación | **Un** modelo sobre **un** dataset grande |
| Ajuste al problema | Cada combinación tiene 6–24 puntos → K-Means local es instantáneo; Spark paraleliza por grupo | Habría que **iterar por combinación** lanzando un job Spark por cada una → miles de jobs, cuello de botella en el driver, overhead >> cómputo |
| K por grupo | Trivial (codo por grupo, dentro de la UDF) | Muy costoso (un modelo distribuido por cada K y cada grupo) |
| Escalado real | Escala por **nº de combinaciones** (particiones), no por filas | Diseñado para escalar por **nº de filas** de un único clustering |
| Dependencia | Requiere `scikit-learn` en los workers | Nativo de Spark |

`pyspark.ml.KMeans` es la herramienta correcta para clusterizar **un** conjunto masivo de puntos de
forma distribuida; **no** para entrenar un modelo por grupo. Nuestro grano es exactamente lo contrario:
paralelismo *embarrassing* de miles de series diminutas. `applyInPandas` (pandas UDF de grouped-map)
envía a cada worker el `pandas.DataFrame` de un grupo y corre sklearn en memoria — el patrón canónico
"un modelo por partición".

**Riesgo y mitigación (disponibilidad de sklearn en Athena Spark).** Athena for Apache Spark permite
añadir librerías Python a la sesión/aplicación; si `scikit-learn` no está instalado, la UDF hace
**fallback a un K-Means 1-D en numpy puro** (Lloyd + k-means++, mismos `random_state`/`n_init`), de modo
que el pipeline nunca se rompe por la ausencia del paquete. El driver imprime un aviso informativo con
la versión detectada. Para instalarlo, añade `scikit-learn` en las propiedades de librerías Python de
la sesión Spark.

---

## 5. Entregable 3 — Tabla consolidada (llaves actuales + columnas nuevas)

`Lista_Outliers_Consolidado_DA` conserva las llaves del script base y añade:

| Grupo | Columnas |
|---|---|
| Llaves actuales | `codmes_val, ventana_meses, codmes, metrica, canal_dsc, flujo_dsc, objetivo_dsc, productos_dsc, dia_semana, dia_semana_dsc, tipo, valor_mes, suma_mes, n_dias, n_meses` |
| Método / K | `metodo` (CLUSTER / IQR_MAD_*), **`k_elegido`** |
| Clusters | `cluster_id, cluster_rank, cluster_center, cluster_n` |
| Fronteras y límites | **`frontera_inf_cluster`, `frontera_sup_cluster`**, `limite_inf, limite_sup`, `limite_inf_iqr, limite_sup_iqr` |
| Percentiles + validación | `q1, mediana, q3, iqr`, **`dif_inf_p25_log`, `dif_sup_p75_log`, `pctl_valida_inf`, `pctl_valida_sup`** |
| Robustos | `mad, mod_zscore`, **`es_outlier`** |

Además, `ESCRIBIR_DIAG=True` escribe `Detalle_Clusters_Outliers_DA` con **todos** los meses (outlier o
no) por combinación, para auditar clusters, fronteras y la validación frontera≈percentil. Ambas tablas
se particionan por `ventana_meses` y se escriben como Parquet en el catálogo Glue, igual que el base.

El uso posterior no cambia: `LEFT ANTI JOIN` de las bandas contra `Lista_Outliers_Consolidado_DA` por
`(ventana_meses, metrica, canal, flujo, objetivo, producto, dia_semana, codmes)`.

---

## 6. Casos borde cubiertos (ver `tests/test_clustering_logic.py`)

- **Caso de aceptación** (191 vs >100 000) → enero **BAJO** por clustering. ✔
- **Pico ALTO aislado** → **ALTO** (doble cola). ✔
- **Distribución continua unimodal** → **0 outliers** (el guard de hueco evita partir el grupo). ✔
- **Bimodal ~50/50** → no se marca (régimen, no anomalía; `MAX_OUTLIER_FRAC`). ✔
- **Pocos meses (<6)** → fallback IQR/MAD. ✔
- **Serie constante** → `IQR_MAD_DEGENERADO`, sin outliers. ✔
- **Valores ≤ 0** → guard `log10(max(v,1e-9))`. ✔

---

## 7. Parámetros clave (`outliers_consolidado_clustering.py`)

`USAR_CLUSTERING, MIN_MESES_CLUSTER=6, K_MAX=5, RANDOM_STATE=42, N_INIT=20, GAP_FACTOR=1.0,`
`MAX_OUTLIER_FRAC=0.40, TOL_PCTL_LOG=0.50, ESCALA_LOG, IQR_FACTOR=1.5, MAD_Z_UMBRAL=3.5, MIN_MESES=4`.
