"""
DTF Fashion — Fine-Tuning con Transferencia de Patrones H&M
=============================================================
Estrategia:
  1. Limpiar y agregar tus 54 ventas en serie diaria
  2. Extraer índices estacionales de H&M (semanal + mensual)
  3. Escalar esos patrones a tu volumen real
  4. Construir forecast híbrido: patrón H&M calibrado con tus datos
  5. Forecast de los próximos 30 días para tu tienda

NO hacemos SARIMA directo sobre 54 puntos (causaría overfitting).
En cambio, usamos H&M como "prior" y tus datos como "corrección".
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.style.use("seaborn-v0_8-whitegrid")

COLORES = {
    "hm":      "#cbd5e1",
    "dtf":     "#2563eb",
    "forecast":"#16a34a",
    "banda":   "#16a34a",
    "alerta":  "#ef4444",
}

# ──────────────────────────────────────────────
# 1. CARGAR Y LIMPIAR TUS DATOS DTF
# ──────────────────────────────────────────────

print("📦 Cargando datos DTF Fashion...")

raw = pd.read_excel("../data/raw/DTF_s_DATA_CORRECT.xlsx", header=None)

# La fila 2 (índice 2) tiene los headers reales
raw.columns = raw.iloc[2]
raw = raw.iloc[3:].reset_index(drop=True)
raw.columns = [
    "idx", "venta_id", "fecha", "estado", "ingresos",
    "cargo", "envio", "total", "diseno", "estado_mx", "pais",
    "categoria", "tipo_prenda"
]

# Limpiar fechas con múltiples formatos en español
def parsear_fecha(texto):
    if pd.isna(texto):
        return pd.NaT
    texto = str(texto).strip().lower()
    texto = (texto
        .replace(" de ", " ")
        .replace("enero","january").replace("febrero","february")
        .replace("marzo","march").replace("abril","april")
        .replace("mayo","may").replace("junio","june")
        .replace("julio","july").replace("agosto","august")
        .replace("septiembre","september").replace("octubre","october")
        .replace("noviembre","november").replace("diciembre","december")
    )
    for fmt in ["%d %B %Y", "%d %B de %Y"]:
        try:
            return pd.to_datetime(texto, format=fmt)
        except:
            pass
    return pd.to_datetime(texto, errors="coerce")

raw["fecha"] = raw["fecha"].apply(parsear_fecha)
raw["ingresos"] = pd.to_numeric(raw["ingresos"], errors="coerce")
raw["total"]    = pd.to_numeric(raw["total"],    errors="coerce")
raw = raw.dropna(subset=["fecha", "ingresos"])

print(f"   ✅ {len(raw)} ventas cargadas")
print(f"   Rango: {raw['fecha'].min().date()} → {raw['fecha'].max().date()}")
print(f"   Ingresos totales: ${raw['total'].sum():,.2f} MXN\n")

# ──────────────────────────────────────────────
# 2. AGREGACIÓN DIARIA DTF
# ──────────────────────────────────────────────

print("📊 Agregando ventas diarias DTF...")

ventas_dia = (
    raw.groupby("fecha")
    .agg(
        unidades=("venta_id", "count"),
        ingresos=("ingresos", "sum"),
        ingreso_neto=("total", "sum"),
    )
    .reset_index()
)

# Serie continua (días sin ventas = 0)
rango = pd.date_range(ventas_dia["fecha"].min(), ventas_dia["fecha"].max(), freq="D")
ventas_dia = (
    ventas_dia.set_index("fecha")
    .reindex(rango)
    .fillna(0)
    .reset_index()
    .rename(columns={"index": "fecha"})
)

print(f"   ✅ {len(ventas_dia)} días en la serie")
print(f"   Días con ventas: {(ventas_dia['unidades'] > 0).sum()}")
print(f"   Días sin ventas: {(ventas_dia['unidades'] == 0).sum()}")
print(f"   Promedio días CON venta: {ventas_dia[ventas_dia['unidades']>0]['unidades'].mean():.1f} unidades")
print(f"   Máximo en un día: {ventas_dia['unidades'].max():.0f} unidades\n")

# ──────────────────────────────────────────────
# 3. ANÁLISIS DE CATEGORÍAS DTF
# ──────────────────────────────────────────────

print("🏆 Top categorías DTF:")
top_cats = (
    raw.groupby("categoria")
    .agg(unidades=("venta_id","count"), ingresos=("ingresos","sum"))
    .sort_values("unidades", ascending=False)
)
for cat, row in top_cats.iterrows():
    pct = row["unidades"] / len(raw) * 100
    print(f"   {cat:<15} {row['unidades']:>3} uds  ({pct:.0f}%)  ${row['ingresos']:>7,.0f}")

print("\n🧥 Top tipos de prenda:")
top_prendas = raw["tipo_prenda"].value_counts()
for prenda, cnt in top_prendas.items():
    pct = cnt / len(raw) * 100
    print(f"   {prenda:<15} {cnt:>3} uds  ({pct:.0f}%)")

# ──────────────────────────────────────────────
# 4. CARGAR PATRONES ESTACIONALES DE H&M
# ──────────────────────────────────────────────

print("\n📦 Cargando patrones estacionales H&M...")

hm = pd.read_csv(
    "../data/processed/hm_ventas_agregadas.csv",
    parse_dates=["fecha"],
    index_col="fecha",
)
hm = hm.sort_index()

# Índice estacional SEMANAL (qué % representa cada día vs promedio)
hm["dia_semana"] = hm.index.dayofweek
idx_semanal = (
    hm.groupby("dia_semana")["total_articulos"].mean()
    / hm["total_articulos"].mean()
)

# Índice estacional MENSUAL
hm["mes"] = hm.index.month
idx_mensual = (
    hm.groupby("mes")["total_articulos"].mean()
    / hm["total_articulos"].mean()
)

print("   Índices estacionales semanales H&M:")
dias = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
for i, (d, v) in enumerate(idx_semanal.items()):
    barra = "█" * int(v * 10)
    print(f"   {dias[i]}: {barra:<15} {v:.3f}")

print("\n   Índices estacionales mensuales H&M:")
meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
for m, v in idx_mensual.items():
    barra = "█" * int(v * 10)
    print(f"   {meses[m-1]}: {barra:<15} {v:.3f}")

# ──────────────────────────────────────────────
# 5. CALIBRACIÓN — Escalar H&M a tu volumen
# ──────────────────────────────────────────────

print("\n⚖️  Calibrando patrones H&M a escala DTF...")

# Promedio real de tu tienda (solo días con ventas para no sesgar)
tu_promedio = ventas_dia[ventas_dia["unidades"] > 0]["unidades"].mean()
print(f"   Tu promedio (días con venta): {tu_promedio:.2f} unidades/día")

# Factor de escala
factor_escala = tu_promedio / hm["total_articulos"].mean()
print(f"   Factor de escala H&M→DTF: {factor_escala:.6f}")

# ──────────────────────────────────────────────
# 6. CORRECCIÓN CON TUS DATOS REALES
# ──────────────────────────────────────────────

print("\n🔧 Calculando corrección con tus datos reales...")

# Para cada mes que tienes datos, calculamos qué tan diferente
# es tu patrón vs el patrón H&M escalado
ventas_dia["mes"] = pd.to_datetime(ventas_dia["fecha"]).dt.month
ventas_dia["dia_semana"] = pd.to_datetime(ventas_dia["fecha"]).dt.dayofweek

# Solo días con ventas para calcular la corrección
con_ventas = ventas_dia[ventas_dia["unidades"] > 0].copy()

# Corrección mensual: tu promedio mensual vs H&M escalado mensual
correccion_mensual = {}
for mes, grupo in con_ventas.groupby("mes"):
    tu_prom_mes     = grupo["unidades"].mean()
    hm_escalado_mes = idx_mensual.get(mes, 1.0) * tu_promedio
    correccion_mensual[mes] = tu_prom_mes / hm_escalado_mes if hm_escalado_mes > 0 else 1.0

print("   Correcciones por mes (1.0 = sin corrección):")
for mes, corr in correccion_mensual.items():
    direccion = "↑" if corr > 1.05 else ("↓" if corr < 0.95 else "≈")
    print(f"   {meses[mes-1]}: {corr:.3f} {direccion}")

# ──────────────────────────────────────────────
# 7. FORECAST 30 DÍAS — MODELO HÍBRIDO CALIBRADO
# ──────────────────────────────────────────────

print("\n🔮 Generando forecast 30 días para DTF Fashion...")

ultima_fecha = ventas_dia["fecha"].max()
fechas_futuro = pd.date_range(ultima_fecha + timedelta(days=1), periods=30)

forecast_rows = []
for fecha in fechas_futuro:
    mes        = fecha.month
    dia_sem    = fecha.dayofweek

    # Patrón H&M escalado
    idx_s = idx_semanal.get(dia_sem, 1.0)
    idx_m = idx_mensual.get(mes, 1.0)
    base  = tu_promedio * idx_s * idx_m

    # Aplicar corrección de tus datos reales
    corr  = correccion_mensual.get(mes, 1.0)
    # Para meses sin datos propios, promediamos las correcciones conocidas
    if mes not in correccion_mensual:
        corr = np.mean(list(correccion_mensual.values()))

    forecast_val = base * corr
    forecast_val = max(forecast_val, 0)

    # Banda de incertidumbre (±30% — conservadora dado el poco historial)
    forecast_rows.append({
        "fecha":    fecha,
        "forecast": round(forecast_val, 2),
        "minimo":   round(forecast_val * 0.70, 2),
        "maximo":   round(forecast_val * 1.30, 2),
        "dia_semana": dias[dia_sem],
        "mes_nombre": meses[mes - 1],
    })

forecast_df = pd.DataFrame(forecast_rows)

# Resumen semanal del forecast
print("\n   Forecast semanal (próximos 30 días):")
forecast_df["semana"] = pd.to_datetime(forecast_df["fecha"]).dt.isocalendar().week
for sem, g in forecast_df.groupby("semana"):
    inicio = g["fecha"].iloc[0].strftime("%d %b")
    fin    = g["fecha"].iloc[-1].strftime("%d %b")
    total  = g["forecast"].sum()
    print(f"   Semana {sem} ({inicio}–{fin}): {total:.1f} unidades estimadas")

# ──────────────────────────────────────────────
# 8. GRÁFICAS
# ──────────────────────────────────────────────

# Gráfica 1 — Serie histórica DTF
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

ax = axes[0]
ax.bar(ventas_dia["fecha"], ventas_dia["unidades"],
       color=COLORES["dtf"], alpha=0.7, width=1)
ax.set_title("Ventas diarias — DTF Fashion (Oct 2025 – Mar 2026)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Unidades vendidas")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.get_xticklabels(), rotation=45)

# Gráfica 2 — Ingresos acumulados
ingresos_acum = ventas_dia["ingresos"].cumsum()
ax2 = axes[1]
ax2.fill_between(ventas_dia["fecha"], ingresos_acum,
                 color=COLORES["dtf"], alpha=0.2)
ax2.plot(ventas_dia["fecha"], ingresos_acum,
         color=COLORES["dtf"], linewidth=2)
ax2.set_title("Ingresos acumulados — DTF Fashion",
              fontsize=13, fontweight="bold")
ax2.set_ylabel("Ingresos acumulados (MXN $)")
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax2.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig("../data/processed/14_dtf_historico.png", dpi=150)
plt.show()
print("\n✅ Gráfica guardada: 14_dtf_historico.png")

# Gráfica 2 — Forecast con banda
fig, ax = plt.subplots()

# Histórico reciente
ax.bar(ventas_dia["fecha"], ventas_dia["unidades"],
       color=COLORES["dtf"], alpha=0.6, width=1, label="Ventas reales")

# Forecast
ax.plot(forecast_df["fecha"], forecast_df["forecast"],
        color=COLORES["forecast"], linewidth=2.5,
        linestyle="--", marker="o", markersize=4,
        label="Forecast 30 días")
ax.fill_between(
    forecast_df["fecha"],
    forecast_df["minimo"],
    forecast_df["maximo"],
    color=COLORES["banda"], alpha=0.15, label="Banda ±30%"
)

ax.axvline(x=ultima_fecha, color="#94a3b8", linestyle=":", linewidth=1.5)
ax.set_title("DTF Fashion — Forecast Próximos 30 Días\n(Patrones H&M calibrados a tu escala)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Unidades vendidas")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("../data/processed/15_dtf_forecast.png", dpi=150)
plt.show()
print("✅ Gráfica guardada: 15_dtf_forecast.png")

# Gráfica 3 — Comparación de estacionalidad H&M vs DTF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Por día de semana
dtf_por_dia = (
    con_ventas.groupby("dia_semana")["unidades"].mean()
    .reindex(range(7))
    .fillna(0)
)
hm_por_dia_norm = idx_semanal * tu_promedio

ax = axes[0]
x = np.arange(7)
ax.bar(x - 0.2, hm_por_dia_norm.values, 0.4,
       color=COLORES["hm"], label="H&M (escalado)")
ax.bar(x + 0.2, dtf_por_dia.values, 0.4,
       color=COLORES["dtf"], alpha=0.8, label="Tu tienda DTF")
ax.set_xticks(x)
ax.set_xticklabels(dias)
ax.set_title("Patrón semanal\nH&M vs DTF", fontweight="bold")
ax.set_ylabel("Unidades promedio")
ax.legend()

# Por mes
dtf_por_mes = (
    con_ventas.groupby("mes")["unidades"].mean()
    .reindex(range(1, 13))
    .fillna(0)
)
hm_por_mes_norm = idx_mensual * tu_promedio

ax = axes[1]
x = np.arange(12)
ax.plot(x, hm_por_mes_norm.values, color=COLORES["hm"],
        marker="o", linewidth=2, label="H&M (escalado)")
ax.plot(x, dtf_por_mes.values, color=COLORES["dtf"],
        marker="s", linewidth=2, label="Tu tienda DTF")
ax.set_xticks(x)
ax.set_xticklabels(meses, rotation=45)
ax.set_title("Patrón mensual\nH&M vs DTF", fontweight="bold")
ax.set_ylabel("Unidades promedio")
ax.legend()

plt.suptitle("Comparación de Estacionalidad — H&M vs DTF Fashion",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../data/processed/16_estacionalidad_comparada.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfica guardada: 16_estacionalidad_comparada.png")

# ──────────────────────────────────────────────
# 9. EXPORTAR
# ──────────────────────────────────────────────

forecast_df.to_csv("../data/processed/dtf_forecast_30dias.csv", index=False)
print("💾 Forecast exportado: dtf_forecast_30dias.csv")

ventas_dia.to_csv("../data/processed/dtf_ventas_diarias.csv", index=False)
print("💾 Histórico DTF exportado: dtf_ventas_diarias.csv")

# ──────────────────────────────────────────────
# 10. RESUMEN FINAL
# ──────────────────────────────────────────────

print("\n" + "=" * 58)
print("  RESUMEN EJECUTIVO — DTF Fashion")
print("=" * 58)
print(f"  Período analizado:     Oct 2025 → Mar 2026")
print(f"  Total ventas:          {len(raw)} unidades")
print(f"  Ingresos totales:      ${raw['ingresos'].sum():>10,.2f} MXN")
print(f"  Ingreso neto total:    ${raw['total'].sum():>10,.2f} MXN")
print(f"  Ticket promedio:       ${raw['ingresos'].mean():>10,.2f} MXN")
print(f"  Categoría #1:          {top_cats.index[0]} ({top_cats.iloc[0]['unidades']} uds)")
print(f"  Prenda #1:             {top_prendas.index[0]} ({top_prendas.iloc[0]} uds)")
print(f"  Forecast próx. 30d:    {forecast_df['forecast'].sum():.0f} unidades estimadas")
print(f"  Rango forecast:        {forecast_df['minimo'].sum():.0f} – {forecast_df['maximo'].sum():.0f} uds")
print("=" * 58)
print("\n✅ Fine-tuning completado. Revisa las 3 gráficas generadas.")