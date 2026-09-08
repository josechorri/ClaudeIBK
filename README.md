# Outliers mensuales con límites dinámicos por clustering

Pipeline PySpark (Amazon Athena for Apache Spark) para detectar meses atípicos por combinación
`(metrica, canal, flujo, objetivo, producto, dia_semana)` y ventana (6/12/18/24 meses), antes de
calcular las bandas Tukey por día de semana.

| Archivo | Qué es |
|---|---|
| `outliers_consolidado.py` | Script **base** (referencia): límites fijos P25/P75 + cerca Tukey IQR/MAD en `log10`. |
| `outliers_consolidado_clustering.py` | **Extensión**: los cortes `limite_inf`/`limite_sup` se derivan de la estructura de clusters (K-Means 1-D) de cada combinación; K automático por método del codo; validación vs percentiles; IQR/MAD como fallback. |
| `DISENO_CLUSTERING.md` | Explicación de diseño: elección de K, derivación de límites, fallback y trade-off `applyInPandas`+sklearn vs `pyspark.ml` KMeans. |
| `tests/test_clustering_logic.py` | Pruebas (sin Spark) del criterio de aceptación y casos borde. |

## Ejecutar las pruebas locales

```bash
pip install numpy pandas scikit-learn
python3 tests/test_clustering_logic.py
```

## Criterio de aceptación

`Venta TC / Marketing Contextual / Ventas / HTML / ENVIOS / Lunes`, enero ≈ 191 envíos con el resto de
meses > 100 000 → el clustering aísla enero como **outlier BAJO** (verificado en las pruebas).
