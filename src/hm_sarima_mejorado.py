"""
H&M Dataset — SARIMA Mejorado con Limpieza de Outliers
=======================================================
Estrategia:
  1. Detectar días anormales (outliers) usando IQR
  2. Suavizarlos con la mediana de días similares
  3. Reentrenar SARIMA con la serie limpia
  4. Comparar MAPE original vs mejorado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
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
# 2. DETECCIÓN DE OUTLIERS (método IQR)
# ──────────────────────────────────────────────

print("🔍 Detectando outliers...")

Q1 = serie.quantile(0.25)
Q3 = serie.quantile(0.75)
IQR = Q3 - Q1

# Umbral: más de 2.5 veces el IQR por encima/abajo
limite_superior = Q3 + 2.5 * IQR
limite_inferior = Q1 - 2.5 * IQR

outliers = serie[(serie > limite_superior) | (serie < limite_inferior)]

print(f"   Umbral superior: {limite_superior:,.0f} artículos")
print(f"   Umbral inferior: {limite_inferior:,.0f} artículos")
print(f"   Outliers encontrados: {len(outliers)} días")
for fecha, val in outliers.items():
    print(f"      {fecha.strftime('%d %b %Y')} — {val:,.0f} artículos "
          f"({'↑ pico' if val > limite_superior else '↓ valle'})")

# ──────────────────────────────────────────────
# 3. SUAVIZADO DE OUTLIERS
# ──────────────────────────────────────────────

print("\n🧹 Suavizando outliers...")
serie_limpia = serie.copy()

for fecha in outliers.index:
    # Reemplazar con la mediana de la misma ventana de 7 días
    inicio = fecha - pd.Timedelta(days=7)
    fin    = fecha + pd.Timedelta(days=7)
    ventana = serie[(serie.index >= inicio) & (serie.index <= fin) & (serie.index != fecha)]
    serie_limpia[fecha] = ventana.median()
    print(f"   {fecha.strftime('%d %b %Y')}: {serie[fecha]:,.0f} → {serie_limpia[fecha]:,.0f}")

# ──────────────────────────────────────────────
# 4. GRÁFICA — Serie original vs limpia
# ──────────────────────────────────────────────

fig, ax = plt.subplots()
ax.plot(serie.index, serie,
        color="#cbd5e1", linewidth=0.8, label="Serie original")
ax.plot(serie_limpia.index, serie_limpia,
        color="#2563eb", linewidth=1, label="Serie limpia")
ax.scatter(outliers.index, outliers.values,
           color="#ef4444", zorder=5, s=40, label=f"{len(outliers)} outliers corregidos")
ax.set_title("Limpieza de outliers — Original vs Suavizado",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Artículos vendidos")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("../data/processed/11_outliers_detectados.png", dpi=150)
plt.show()
print("\n✅ Gráfica guardada: 11_outliers_detectados.png")

# ──────────────────────────────────────────────
# 5. DIVISIÓN TRAIN / TEST
# ──────────────────────────────────────────────

DIAS_TEST = 30
train_orig  = serie.iloc[:-DIAS_TEST]
train_limpio = serie_limpia.iloc[:-DIAS_TEST]
test         = serie.iloc[-DIAS_TEST:]  # test siempre con datos reales

# ──────────────────────────────────────────────
# 6. ENTRENAMIENTO — SARIMA ORIGINAL (referencia)
# ──────────────────────────────────────────────

print("\n⚙️  Entrenando SARIMA original (con outliers)...")
print("   (2-5 minutos...)\n")

sarima_orig = SARIMAX(
    train_orig,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

pred_orig = sarima_orig.forecast(steps=DIAS_TEST)
pred_orig.index = test.index
mape_orig = (np.abs((test - pred_orig) / test) * 100).mean()
mae_orig  = mean_absolute_error(test, pred_orig)
print(f"   ✅ SARIMA original — MAPE: {mape_orig:.2f}%")

# ──────────────────────────────────────────────
# 7. ENTRENAMIENTO — SARIMA MEJORADO (sin outliers)
# ──────────────────────────────────────────────

print("\n⚙️  Entrenando SARIMA mejorado (sin outliers)...")
print("   (2-5 minutos...)\n")

sarima_limpio = SARIMAX(
    train_limpio,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

pred_limpio = sarima_limpio.forecast(steps=DIAS_TEST)
pred_limpio.index = test.index
mape_limpio = (np.abs((test - pred_limpio) / test) * 100).mean()
mae_limpio  = mean_absolute_error(test, pred_limpio)
print(f"   ✅ SARIMA mejorado — MAPE: {mape_limpio:.2f}%")

# ──────────────────────────────────────────────
# 8. COMPARACIÓN DE RESULTADOS
# ──────────────────────────────────────────────

mejora = mape_orig - mape_limpio
print("\n" + "=" * 55)
print("  COMPARACIÓN SARIMA ORIGINAL vs MEJORADO")
print("=" * 55)
print(f"  {'Métrica':<30} {'Original':>10} {'Mejorado':>10}")
print(f"  {'-'*50}")
print(f"  {'MAE  (artículos)':<30} {mae_orig:>10,.0f} {mae_limpio:>10,.0f}")
print(f"  {'MAPE (porcentaje)':<30} {mape_orig:>9.2f}% {mape_limpio:>9.2f}%")
print(f"  {'Outliers eliminados':<30} {'—':>10} {len(outliers):>10}")
print("=" * 55)
if mejora > 0:
    print(f"\n  ✅ Mejora: {mejora:.2f} puntos porcentuales menos de error")
else:
    print(f"\n  ℹ️  Diferencia: {mejora:.2f}pp (el modelo ya era robusto)")

# ──────────────────────────────────────────────
# 9. GRÁFICA — Comparación predicciones
# ──────────────────────────────────────────────

fig, ax = plt.subplots()
ax.plot(train_orig.iloc[-45:].index, train_orig.iloc[-45:],
        color="#cbd5e1", linewidth=1, label="Histórico")
ax.plot(test.index, test,
        color="#2563eb", linewidth=2, label="Real")
ax.plot(pred_orig.index, pred_orig,
        color="#f59e0b", linewidth=1.8, linestyle="--",
        label=f"SARIMA original  (MAPE {mape_orig:.1f}%)")
ax.plot(pred_limpio.index, pred_limpio,
        color="#16a34a", linewidth=2,
        label=f"SARIMA mejorado  (MAPE {mape_limpio:.1f}%)")

ax.axvline(x=test.index[0], color="#94a3b8", linestyle=":", linewidth=1.5)
ax.set_title("SARIMA Original vs Mejorado (sin outliers)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Artículos vendidos")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("../data/processed/12_sarima_original_vs_mejorado.png", dpi=150)
plt.show()
print("✅ Gráfica guardada: 12_sarima_original_vs_mejorado.png")

# ──────────────────────────────────────────────
# 10. FORECAST FUTURO 30 DÍAS con modelo mejorado
# ──────────────────────────────────────────────

print("\n🔮 Generando forecast 30 días con modelo mejorado...")

sarima_final = SARIMAX(
    serie_limpia,  # toda la serie limpia
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

fechas_futuro  = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=30)
forecast       = sarima_final.get_forecast(steps=30)
forecast_mean  = forecast.predicted_mean
forecast_ci    = forecast.conf_int(alpha=0.10)  # intervalo 90% (más conservador)
forecast_mean.index = fechas_futuro
forecast_ci.index   = fechas_futuro

# Clamp: no puede ser negativo
forecast_mean = forecast_mean.clip(lower=0)
forecast_ci   = forecast_ci.clip(lower=0)

fig, ax = plt.subplots()
ax.plot(serie_limpia.iloc[-60:].index, serie_limpia.iloc[-60:],
        color="#2563eb", linewidth=1.5, label="Histórico reciente (limpio)")
ax.plot(forecast_mean.index, forecast_mean,
        color="#16a34a", linewidth=2.5, linestyle="--",
        label="Forecast 30 días")
ax.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="#16a34a", alpha=0.15, label="Intervalo confianza 90%"
)
ax.axvline(x=serie.index[-1], color="#94a3b8", linestyle=":", linewidth=1.5)
ax.set_title(f"Forecast SARIMA Mejorado — Próximos 30 días\nMAPE: {mape_limpio:.2f}%",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Artículos vendidos")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("../data/processed/13_forecast_mejorado.png", dpi=150)
plt.show()
print("✅ Gráfica guardada: 13_forecast_mejorado.png")

# Exportar
forecast_df = pd.DataFrame({
    "fecha":           forecast_mean.index,
    "forecast":        forecast_mean.values.round(0).astype(int),
    "limite_inferior": forecast_ci.iloc[:, 0].values.round(0).astype(int),
    "limite_superior": forecast_ci.iloc[:, 1].values.round(0).astype(int),
})
forecast_df.to_csv("../data/processed/hm_forecast_final.csv", index=False)
print("💾 Forecast exportado: hm_forecast_final.csv")

print("\n" + "=" * 55)
print("  🎉 Pipeline completo")
print(f"  MAPE final: {mape_limpio:.2f}%")
print(f"  Outliers corregidos: {len(outliers)}")
print("=" * 55)