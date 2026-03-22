"""
H&M Dataset — Modelo SARIMA
============================
Construye, entrena y evalúa un modelo SARIMA(1,1,1)(1,1,1)[7]
para forecasting de ventas diarias.

Parámetros elegidos con base en el análisis ACF/PACF:
  - p=1, d=1, q=1  (parte no estacional)
  - P=1, D=1, Q=1  (parte estacional)
  - s=7            (ciclo semanal)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.style.use("seaborn-v0_8-whitegrid")

# ──────────────────────────────────────────────
# 1. CARGA Y PREPARACIÓN
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
# 2. PRUEBA DE ESTACIONARIEDAD (Dickey-Fuller)
# ──────────────────────────────────────────────

print("🔬 Prueba de estacionariedad (ADF)...")
resultado_adf = adfuller(serie)
print(f"   Estadístico ADF: {resultado_adf[0]:.4f}")
print(f"   p-value:         {resultado_adf[1]:.4f}")
if resultado_adf[1] < 0.05:
    print("   ✅ La serie ES estacionaria (p < 0.05)")
else:
    print("   ⚠️  La serie NO es estacionaria → se aplicará d=1 en SARIMA")

# ──────────────────────────────────────────────
# 3. DIVISIÓN TRAIN / TEST
# ──────────────────────────────────────────────

# Usamos los últimos 30 días como test
DIAS_TEST = 30
train = serie.iloc[:-DIAS_TEST]
test  = serie.iloc[-DIAS_TEST:]

print(f"\n📅 División de datos:")
print(f"   Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} días)")
print(f"   Test:  {test.index[0].date()}  → {test.index[-1].date()}  ({len(test)} días)")

# ──────────────────────────────────────────────
# 4. ENTRENAMIENTO SARIMA
# ──────────────────────────────────────────────

print("\n⚙️  Entrenando SARIMA(1,1,1)(1,1,1)[7]...")
print("   (Esto puede tardar 2-5 minutos, es normal...)\n")

modelo = SARIMAX(
    train,
    order=(1, 1, 1),           # p, d, q
    seasonal_order=(1, 1, 1, 7),  # P, D, Q, s
    enforce_stationarity=False,
    enforce_invertibility=False,
)

resultado = modelo.fit(disp=False)
print(resultado.summary())

# ──────────────────────────────────────────────
# 5. PREDICCIÓN EN TEST
# ──────────────────────────────────────────────

print("\n📈 Generando predicciones...")
pred = resultado.forecast(steps=DIAS_TEST)
pred.index = test.index

# Métricas
mae  = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))
mape = (np.abs((test - pred) / test) * 100).mean()

print(f"\n📊 Métricas de evaluación (últimos {DIAS_TEST} días):")
print(f"   MAE  (Error absoluto medio):     {mae:>10,.0f} artículos")
print(f"   RMSE (Raíz error cuadrático):    {rmse:>10,.0f} artículos")
print(f"   MAPE (Error porcentual medio):   {mape:>10.2f}%")

if mape < 15:
    print("   ✅ Modelo BUENO — MAPE menor al 15%")
elif mape < 25:
    print("   ⚠️  Modelo ACEPTABLE — MAPE entre 15-25%")
else:
    print("   ❌ Modelo MEJORABLE — MAPE mayor al 25% → XGBoost ayudará")

# ──────────────────────────────────────────────
# 6. GRÁFICA — Real vs Predicho
# ──────────────────────────────────────────────

fig, ax = plt.subplots()

# Últimos 60 días de train para contexto
ax.plot(train.iloc[-60:].index, train.iloc[-60:],
        color="#cbd5e1", linewidth=1, label="Histórico")
ax.plot(test.index, test,
        color="#2563eb", linewidth=2, label="Real")
ax.plot(pred.index, pred,
        color="#f59e0b", linewidth=2, linestyle="--", label="SARIMA predicción")

ax.axvline(x=test.index[0], color="#94a3b8", linestyle=":", linewidth=1.5)
ax.text(test.index[0], ax.get_ylim()[1]*0.95, " inicio test",
        color="#64748b", fontsize=9)

ax.set_title(f"SARIMA(1,1,1)(1,1,1)[7] — Real vs Predicho\nMAPE: {mape:.2f}%",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Artículos vendidos")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("../data/processed/06_sarima_prediccion.png", dpi=150)
plt.show()
print("\n✅ Gráfica guardada: 06_sarima_prediccion.png")

# ──────────────────────────────────────────────
# 7. FORECAST FUTURO — Próximos 30 días
# ──────────────────────────────────────────────

print("\n🔮 Generando forecast de los próximos 30 días...")

modelo_final = SARIMAX(
    serie,  # ahora entrenamos con TODOS los datos
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

forecast = modelo_final.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean
forecast_ci   = forecast.conf_int(alpha=0.05)  # intervalo 95%

# Gráfica del forecast futuro
fig, ax = plt.subplots()
ax.plot(serie.iloc[-60:].index, serie.iloc[-60:],
        color="#2563eb", linewidth=1.5, label="Histórico reciente")
ax.plot(forecast_mean.index, forecast_mean,
        color="#f59e0b", linewidth=2, linestyle="--", label="Forecast 30 días")
ax.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="#f59e0b", alpha=0.15, label="Intervalo de confianza 95%"
)
ax.axvline(x=serie.index[-1], color="#94a3b8", linestyle=":", linewidth=1.5)
ax.set_title("Forecast — Próximos 30 días", fontsize=13, fontweight="bold")
ax.set_ylabel("Artículos vendidos")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("../data/processed/07_sarima_forecast.png", dpi=150)
plt.show()
print("✅ Gráfica guardada: 07_sarima_forecast.png")

# ──────────────────────────────────────────────
# 8. EXPORTAR FORECAST
# ──────────────────────────────────────────────

forecast_df = pd.DataFrame({
    "fecha":          forecast_mean.index,
    "forecast":       forecast_mean.values.round(0).astype(int),
    "limite_inferior": forecast_ci.iloc[:, 0].values.round(0).astype(int),
    "limite_superior": forecast_ci.iloc[:, 1].values.round(0).astype(int),
})
forecast_df.to_csv("../data/processed/hm_forecast_sarima.csv", index=False)

print("\n💾 Forecast exportado: hm_forecast_sarima.csv")
print("\n" + "="*55)
print("  ✅ SARIMA completado")
print(f"  MAPE final: {mape:.2f}%")
print("  Siguiente paso: agregar XGBoost para mejorar el modelo")
print("="*55)