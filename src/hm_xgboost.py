"""
H&M Dataset — Modelo Híbrido SARIMA + XGBoost
===============================================
XGBoost aprende de los ERRORES que SARIMA comete.
Predicción final = predicción SARIMA + corrección XGBoost

Features que usará XGBoost:
  - Día de la semana
  - Mes
  - Semana del año
  - Es fin de semana
  - Rezagos (ventas de días anteriores)
  - Error rezagado de SARIMA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.style.use("seaborn-v0_8-whitegrid")

# ──────────────────────────────────────────────
# 1. CARGA
# ──────────────────────────────────────────────

print("📦 Cargando datos...")
df = pd.read_csv(
    "../data/processed/hm_ventas_agregadas.csv",
    parse_dates=["fecha"],
    index_col="fecha",
)
df = df.sort_index()
serie = df["total_articulos"].astype(float)
print(f"   ✅ {len(serie)} días cargados\n")

# ──────────────────────────────────────────────
# 2. DIVISIÓN TRAIN / TEST
# ──────────────────────────────────────────────

DIAS_TEST = 30
train = serie.iloc[:-DIAS_TEST]
test  = serie.iloc[-DIAS_TEST:]

print(f"📅 Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} días)")
print(f"📅 Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} días)\n")

# ──────────────────────────────────────────────
# 3. ENTRENAMIENTO SARIMA (base)
# ──────────────────────────────────────────────

print("⚙️  Entrenando SARIMA(1,1,1)(1,1,1)[7]...")
print("   (2-5 minutos, es normal...)\n")

sarima = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

# Predicciones SARIMA en train (para calcular residuos)
pred_train_sarima = sarima.fittedvalues
residuos_train    = train - pred_train_sarima

# Predicciones SARIMA en test
pred_test_sarima  = sarima.forecast(steps=DIAS_TEST)
pred_test_sarima.index = test.index

mape_sarima = (np.abs((test - pred_test_sarima) / test) * 100).mean()
print(f"   ✅ SARIMA listo — MAPE: {mape_sarima:.2f}%\n")

# ──────────────────────────────────────────────
# 4. CONSTRUCCIÓN DE FEATURES PARA XGBOOST
# ──────────────────────────────────────────────

def construir_features(serie_idx, serie_vals=None, residuos=None):
    """
    Construye el DataFrame de features a partir de un índice de fechas.
    """
    feats = pd.DataFrame(index=serie_idx)

    # Features de calendario
    feats["dia_semana"]    = serie_idx.dayofweek          # 0=Lun, 6=Dom
    feats["mes"]           = serie_idx.month
    feats["semana_anio"]   = serie_idx.isocalendar().week.astype(int)
    feats["es_finde"]      = (serie_idx.dayofweek >= 5).astype(int)
    feats["dia_mes"]       = serie_idx.day
    feats["trimestre"]     = serie_idx.quarter

    # Rezagos de ventas reales (si están disponibles)
    if serie_vals is not None:
        s = pd.Series(serie_vals, index=serie_idx)
        for lag in [1, 2, 3, 7, 14]:
            feats[f"lag_{lag}"] = s.shift(lag).values

    # Rezago del error SARIMA
    if residuos is not None:
        r = pd.Series(residuos.values, index=residuos.index)
        feats["error_lag1"] = r.reindex(serie_idx).shift(1).values
        feats["error_lag7"] = r.reindex(serie_idx).shift(7).values

    return feats.fillna(0)


print("🔧 Construyendo features...")

# Features de entrenamiento
X_train = construir_features(train.index, train.values, residuos_train)
y_train = residuos_train.values  # XGBoost aprende a predecir los RESIDUOS

# Features de test (usamos los últimos valores conocidos para los rezagos)
serie_completa = pd.concat([train, test])
X_test = construir_features(test.index, serie_completa.reindex(test.index).values, residuos_train)

print(f"   Features usados: {list(X_train.columns)}\n")

# ──────────────────────────────────────────────
# 5. ENTRENAMIENTO XGBOOST
# ──────────────────────────────────────────────

print("🚀 Entrenando XGBoost...")

modelo_xgb = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
modelo_xgb.fit(X_train, y_train)
print("   ✅ XGBoost listo\n")

# ──────────────────────────────────────────────
# 6. PREDICCIÓN HÍBRIDA
# ──────────────────────────────────────────────

correccion_xgb  = modelo_xgb.predict(X_test)
pred_hibrida    = pred_test_sarima.values + correccion_xgb
pred_hibrida    = np.maximum(pred_hibrida, 0)  # no puede ser negativo

pred_hibrida_s  = pd.Series(pred_hibrida, index=test.index)

# ──────────────────────────────────────────────
# 7. COMPARACIÓN DE MÉTRICAS
# ──────────────────────────────────────────────

mae_sarima  = mean_absolute_error(test, pred_test_sarima)
mae_hibrido = mean_absolute_error(test, pred_hibrida_s)

mape_hibrido = (np.abs((test - pred_hibrida_s) / test) * 100).mean()
rmse_hibrido = np.sqrt(mean_squared_error(test, pred_hibrida_s))

print("=" * 55)
print("  COMPARACIÓN DE MODELOS")
print("=" * 55)
print(f"  {'Métrica':<30} {'SARIMA':>8} {'Híbrido':>8}")
print(f"  {'-'*46}")
print(f"  {'MAE  (artículos)':<30} {mae_sarima:>8,.0f} {mae_hibrido:>8,.0f}")
print(f"  {'MAPE (porcentaje)':<30} {mape_sarima:>7.2f}% {mape_hibrido:>7.2f}%")
print("=" * 55)

mejora = mape_sarima - mape_hibrido
if mejora > 0:
    print(f"\n  ✅ XGBoost mejoró el MAPE en {mejora:.2f} puntos porcentuales")
else:
    print(f"\n  ℹ️  SARIMA ya era sólido; diferencia: {mejora:.2f}pp")

# ──────────────────────────────────────────────
# 8. GRÁFICA — Comparación SARIMA vs Híbrido
# ──────────────────────────────────────────────

fig, ax = plt.subplots()
ax.plot(train.iloc[-45:].index, train.iloc[-45:],
        color="#cbd5e1", linewidth=1, label="Histórico")
ax.plot(test.index, test,
        color="#2563eb", linewidth=2, label="Real")
ax.plot(test.index, pred_test_sarima,
        color="#f59e0b", linewidth=1.8, linestyle="--",
        label=f"SARIMA  (MAPE {mape_sarima:.1f}%)")
ax.plot(test.index, pred_hibrida_s,
        color="#16a34a", linewidth=2,
        label=f"SARIMA+XGBoost  (MAPE {mape_hibrido:.1f}%)")

ax.axvline(x=test.index[0], color="#94a3b8", linestyle=":", linewidth=1.5)
ax.set_title("Modelo Híbrido SARIMA + XGBoost\nReal vs SARIMA vs Híbrido",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Artículos vendidos")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("../data/processed/08_hibrido_comparacion.png", dpi=150)
plt.show()
print("\n✅ Gráfica guardada: 08_hibrido_comparacion.png")

# ──────────────────────────────────────────────
# 9. IMPORTANCIA DE FEATURES
# ──────────────────────────────────────────────

importancias = pd.Series(
    modelo_xgb.feature_importances_,
    index=X_train.columns
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
importancias.plot(kind="barh", ax=ax, color="#2563eb", edgecolor="white")
ax.set_title("XGBoost — Importancia de features", fontsize=13, fontweight="bold")
ax.set_xlabel("Importancia")
plt.tight_layout()
plt.savefig("../data/processed/09_feature_importance.png", dpi=150)
plt.show()
print("✅ Gráfica guardada: 09_feature_importance.png")

# ──────────────────────────────────────────────
# 10. FORECAST FUTURO 30 DÍAS (híbrido)
# ──────────────────────────────────────────────

print("\n🔮 Generando forecast híbrido 30 días...")

modelo_sarima_final = SARIMAX(
    serie,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

fechas_futuro     = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=30)
sarima_futuro     = modelo_sarima_final.forecast(steps=30)
sarima_futuro.index = fechas_futuro

X_futuro = construir_features(fechas_futuro, serie.iloc[-14:].reindex(fechas_futuro).values, residuos_train)
correccion_futuro = modelo_xgb.predict(X_futuro)
forecast_hibrido  = pd.Series(
    np.maximum(sarima_futuro.values + correccion_futuro, 0),
    index=fechas_futuro,
)

# Gráfica forecast futuro
fig, ax = plt.subplots()
ax.plot(serie.iloc[-60:].index, serie.iloc[-60:],
        color="#2563eb", linewidth=1.5, label="Histórico reciente")
ax.plot(forecast_hibrido.index, forecast_hibrido,
        color="#16a34a", linewidth=2.5, linestyle="--",
        label="Forecast híbrido 30 días")
ax.fill_between(
    forecast_hibrido.index,
    forecast_hibrido * 0.85,
    forecast_hibrido * 1.15,
    color="#16a34a", alpha=0.12, label="Banda ±15%"
)
ax.axvline(x=serie.index[-1], color="#94a3b8", linestyle=":", linewidth=1.5)
ax.set_title("Forecast Híbrido SARIMA+XGBoost — Próximos 30 días",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Artículos vendidos")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("../data/processed/10_forecast_hibrido.png", dpi=150)
plt.show()
print("✅ Gráfica guardada: 10_forecast_hibrido.png")

# Exportar
forecast_hibrido.to_frame("forecast_articulos").to_csv(
    "../data/processed/hm_forecast_hibrido.csv"
)
print("💾 Forecast exportado: hm_forecast_hibrido.csv")

print("\n" + "="*55)
print("  🎉 Modelo híbrido SARIMA + XGBoost completado")
print(f"  MAPE final: {mape_hibrido:.2f}%")
print("="*55)