"""
H&M Dataset — Exploración Visual de la Serie de Tiempo
=======================================================
Antes de modelar con SARIMA, necesitamos entender:
1. Cómo se ven las ventas diarias
2. Si hay tendencia (sube o baja con el tiempo)
3. Si hay estacionalidad (patrones que se repiten)
4. Si hay outliers (días raros que distorsionan el modelo)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ──────────────────────────────────────────────
# CONFIGURACIÓN VISUAL
# ──────────────────────────────────────────────

plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.style.use("seaborn-v0_8-whitegrid")

# ──────────────────────────────────────────────
# CARGA
# ──────────────────────────────────────────────

print("📦 Cargando datos...")
df = pd.read_csv(
    "../data/processed/hm_ventas_agregadas.csv",
    parse_dates=["fecha"],
)
df = df.set_index("fecha").sort_index()
print(f"   ✅ {len(df)} días cargados\n")

# ──────────────────────────────────────────────
# GRÁFICA 1 — Serie completa
# ──────────────────────────────────────────────

fig, ax = plt.subplots()
ax.plot(df.index, df["total_articulos"], color="#2563eb", linewidth=0.8, alpha=0.9)
ax.set_title("Ventas diarias H&M — Serie completa", fontsize=14, fontweight="bold")
ax.set_ylabel("Artículos vendidos")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../data/processed/01_serie_completa.png", dpi=150)
plt.show()
print("✅ Gráfica 1 guardada: 01_serie_completa.png")

# ──────────────────────────────────────────────
# GRÁFICA 2 — Promedio por día de la semana
# ──────────────────────────────────────────────

df["dia_semana"] = df.index.day_name()
orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
nombres_dias = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]

promedio_dia = (
    df.groupby("dia_semana")["total_articulos"]
    .mean()
    .reindex(orden_dias)
)

fig, ax = plt.subplots()
colores = ["#93c5fd"] * 5 + ["#2563eb"] * 2  # fines de semana más oscuros
bars = ax.bar(nombres_dias, promedio_dia.values, color=colores, edgecolor="white")
ax.set_title("Promedio de ventas por día de la semana", fontsize=14, fontweight="bold")
ax.set_ylabel("Artículos vendidos (promedio)")
for bar, val in zip(bars, promedio_dia.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("../data/processed/02_por_dia_semana.png", dpi=150)
plt.show()
print("✅ Gráfica 2 guardada: 02_por_dia_semana.png")

# ──────────────────────────────────────────────
# GRÁFICA 3 — Promedio por mes
# ──────────────────────────────────────────────

df["mes"] = df.index.month
orden_meses = list(range(1, 13))
nombres_meses = ["Ene","Feb","Mar","Abr","May","Jun",
                 "Jul","Ago","Sep","Oct","Nov","Dic"]

promedio_mes = df.groupby("mes")["total_articulos"].mean().reindex(orden_meses)

fig, ax = plt.subplots()
ax.plot(nombres_meses, promedio_mes.values, marker="o", color="#2563eb",
        linewidth=2, markersize=7)
ax.fill_between(range(12), promedio_mes.values, alpha=0.1, color="#2563eb")
ax.set_title("Estacionalidad mensual — Promedio de ventas por mes", fontsize=14, fontweight="bold")
ax.set_ylabel("Artículos vendidos (promedio)")
ax.set_xticks(range(12))
ax.set_xticklabels(nombres_meses)
plt.tight_layout()
plt.savefig("../data/processed/03_estacionalidad_mensual.png", dpi=150)
plt.show()
print("✅ Gráfica 3 guardada: 03_estacionalidad_mensual.png")

# ──────────────────────────────────────────────
# GRÁFICA 4 — Media móvil (tendencia)
# ──────────────────────────────────────────────

df["media_7d"]  = df["total_articulos"].rolling(7).mean()
df["media_30d"] = df["total_articulos"].rolling(30).mean()

fig, ax = plt.subplots()
ax.plot(df.index, df["total_articulos"], color="#cbd5e1",
        linewidth=0.6, alpha=0.8, label="Diario")
ax.plot(df.index, df["media_7d"],  color="#f59e0b",
        linewidth=1.5, label="Media 7 días")
ax.plot(df.index, df["media_30d"], color="#2563eb",
        linewidth=2,   label="Media 30 días (tendencia)")
ax.set_title("Tendencia de ventas con medias móviles", fontsize=14, fontweight="bold")
ax.set_ylabel("Artículos vendidos")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("../data/processed/04_tendencia.png", dpi=150)
plt.show()
print("✅ Gráfica 4 guardada: 04_tendencia.png")

# ──────────────────────────────────────────────
# GRÁFICA 5 — ACF y PACF (para elegir parámetros SARIMA)
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df["total_articulos"].dropna(),  lags=60, ax=axes[0], color="#2563eb")
plot_pacf(df["total_articulos"].dropna(), lags=60, ax=axes[1], color="#2563eb")
axes[0].set_title("ACF — Autocorrelación", fontweight="bold")
axes[1].set_title("PACF — Autocorrelación parcial", fontweight="bold")
plt.suptitle("Diagnóstico para parámetros SARIMA (p, d, q)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("../data/processed/05_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfica 5 guardada: 05_acf_pacf.png")

# ──────────────────────────────────────────────
# RESUMEN ESTADÍSTICO
# ──────────────────────────────────────────────

print("\n" + "="*50)
print("  RESUMEN ESTADÍSTICO")
print("="*50)
stats = df["total_articulos"].describe()
print(f"  Promedio diario:  {stats['mean']:>10,.0f} artículos")
print(f"  Mediana diaria:   {stats['50%']:>10,.0f} artículos")
print(f"  Día más bajo:     {stats['min']:>10,.0f} artículos")
print(f"  Día más alto:     {stats['max']:>10,.0f} artículos")
print(f"  Desv. estándar:   {stats['std']:>10,.0f} artículos")

dia_max = df["total_articulos"].idxmax()
dia_min = df["total_articulos"].idxmin()
print(f"\n  📈 Pico máximo:   {dia_max.strftime('%d %b %Y')} "
      f"({df.loc[dia_max, 'total_articulos']:,} uds)")
print(f"  📉 Día mínimo:    {dia_min.strftime('%d %b %Y')} "
      f"({df.loc[dia_min, 'total_articulos']:,} uds)")
print("="*50)
print("\n✅ Exploración completada. Revisa las 5 gráficas generadas.")