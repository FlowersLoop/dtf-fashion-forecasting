"""
H&M Dataset — Limpieza, Agregación y Análisis de Categorías
============================================================
Prepara los datos para un modelo SARIMA + XGBoost de forecasting de ventas.

Estructura de carpetas recomendada:
    hm_proyecto/
    ├── data/
    │   ├── raw/          ← Pon aquí los CSV originales
    │   └── processed/    ← Aquí se guardarán los outputs
    ├── notebooks/        ← Exploración y experimentación
    ├── src/              ← Este script y módulos
    └── models/           ← Modelos entrenados (.pkl, .json)

Uso:
    python hm_limpieza_agregacion.py
    python hm_limpieza_agregacion.py --meses 24
    python hm_limpieza_agregacion.py --meses 12 --data_dir ./mis_datos
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Limpieza y agregación del dataset H&M")
    parser.add_argument("--meses", type=int, default=24, choices=[12, 24],
                        help="Últimos N meses a conservar (12 o 24)")
    parser.add_argument("--data_dir", type=str, default="./data/raw",
                        help="Directorio con los CSV originales")
    parser.add_argument("--output_dir", type=str, default="./data/processed",
                        help="Directorio de salida")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Top N categorías más vendidas")
    return parser.parse_args()


# ──────────────────────────────────────────────
# 1. CARGA EFICIENTE CON TIPOS OPTIMIZADOS
# ──────────────────────────────────────────────

def cargar_transacciones(filepath: str) -> pd.DataFrame:
    """
    Carga transactions_train.csv con dtypes optimizados.
    Reducción típica de RAM: ~60-70% vs carga por defecto.
    """
    print("📦 Cargando transacciones con tipos optimizados...")
    t0 = time.time()

    dtypes = {
        "customer_id":  "category",   # string hash → category ahorra mucho
        "article_id":   "int32",       # 10 dígitos caben en int32
        "price":        "float32",     # float64 → float32
        "sales_channel_id": "int8",    # solo valores 1 y 2
    }

    # t_dat se parsea aparte para obtener datetime
    df = pd.read_csv(
        filepath,
        dtype=dtypes,
        parse_dates=["t_dat"],
        date_format="%Y-%m-%d",     # evita inferencia lenta
    )

    print(f"   ✅ {len(df):,} filas cargadas en {time.time()-t0:.1f}s")
    print(f"   💾 RAM usada: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def cargar_articulos(filepath: str) -> pd.DataFrame:
    """
    Carga articles.csv conservando solo las columnas útiles para el análisis.
    """
    print("\n📦 Cargando artículos...")

    cols_utiles = [
        "article_id",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "section_name",
        "garment_group_name",
        "index_group_name",   # ← categoría de alto nivel (e.g. "Ladieswear")
    ]

    dtypes = {
        "article_id":               "int32",
        "product_type_name":        "category",
        "product_group_name":       "category",
        "graphical_appearance_name":"category",
        "colour_group_name":        "category",
        "section_name":             "category",
        "garment_group_name":       "category",
        "index_group_name":         "category",
    }

    df = pd.read_csv(filepath, usecols=cols_utiles, dtype=dtypes)
    print(f"   ✅ {len(df):,} artículos cargados")
    return df


def cargar_clientes(filepath: str) -> pd.DataFrame:
    """
    Carga customers.csv — opcional para enriquecer, útil para segmentación futura.
    """
    print("\n📦 Cargando clientes...")

    cols_utiles = ["customer_id", "age", "club_member_status", "fashion_news_frequency"]
    dtypes = {
        "customer_id":            "category",
        "age":                    "float32",
        "club_member_status":     "category",
        "fashion_news_frequency": "category",
    }

    df = pd.read_csv(filepath, usecols=cols_utiles, dtype=dtypes)
    print(f"   ✅ {len(df):,} clientes cargados")
    return df


# ──────────────────────────────────────────────
# 2. FILTRADO TEMPORAL
# ──────────────────────────────────────────────

def filtrar_ultimos_meses(df: pd.DataFrame, meses: int) -> pd.DataFrame:
    """
    Conserva solo los últimos N meses de transacciones.
    """
    fecha_max = df["t_dat"].max()
    fecha_corte = fecha_max - pd.DateOffset(months=meses)

    df_filtrado = df[df["t_dat"] > fecha_corte].copy()

    print(f"\n📅 Filtrado temporal:")
    print(f"   Rango original:  {df['t_dat'].min().date()} → {fecha_max.date()}")
    print(f"   Fecha de corte:  {fecha_corte.date()} (últimos {meses} meses)")
    print(f"   Filas conservadas: {len(df_filtrado):,} / {len(df):,} "
          f"({100*len(df_filtrado)/len(df):.1f}%)")
    return df_filtrado


# ──────────────────────────────────────────────
# 3. AGREGACIÓN DIARIA DE VENTAS
# ──────────────────────────────────────────────

def agregar_ventas_diarias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por día sumando:
      - cantidad de artículos vendidos (conteo de filas)
      - ingresos totales (suma de price)
    Rellena días sin ventas con 0 para que SARIMA tenga una serie continua.
    """
    print("\n📊 Agregando ventas diarias...")

    ventas = (
        df.groupby("t_dat")
        .agg(
            total_articulos=("article_id", "count"),
            ingresos_totales=("price", "sum"),
        )
        .reset_index()
        .rename(columns={"t_dat": "fecha"})
    )

    # Rellenar días sin ventas (weekends, festivos, etc.)
    rango_completo = pd.date_range(ventas["fecha"].min(), ventas["fecha"].max(), freq="D")
    ventas = (
        ventas.set_index("fecha")
        .reindex(rango_completo)
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "fecha"})
    )

    ventas["total_articulos"] = ventas["total_articulos"].astype("int32")
    ventas["ingresos_totales"] = ventas["ingresos_totales"].astype("float32")

    print(f"   ✅ Serie diaria: {len(ventas)} días "
          f"({ventas['fecha'].min().date()} → {ventas['fecha'].max().date()})")
    print(f"   📈 Promedio diario: {ventas['total_articulos'].mean():,.0f} artículos")
    print(f"   📈 Pico diario:     {ventas['total_articulos'].max():,} artículos")
    return ventas


# ──────────────────────────────────────────────
# 4. ANÁLISIS DE CATEGORÍAS TOP
# ──────────────────────────────────────────────

def top_categorias(
    df_trans: pd.DataFrame,
    df_art: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identifica las N categorías más vendidas fusionando transacciones con artículos.
    Analiza a tres niveles: grupo de producto, grupo de prenda y sección.
    """
    print(f"\n🏆 Calculando top {top_n} categorías...")

    # Merge eficiente usando int32 en ambos lados
    merged = df_trans[["article_id"]].merge(
        df_art[["article_id", "product_group_name", "garment_group_name",
                "index_group_name", "section_name"]],
        on="article_id",
        how="left",
    )

    resultados = {}

    for col, label in [
        ("product_group_name", "Grupo de Producto"),
        ("garment_group_name", "Grupo de Prenda"),
        ("index_group_name",   "Línea (alto nivel)"),
    ]:
        top = (
            merged[col]
            .value_counts()
            .head(top_n)
            .reset_index()
            .rename(columns={"count": "unidades_vendidas", col: "categoria"})
        )
        top["porcentaje"] = (top["unidades_vendidas"] / len(merged) * 100).round(2)
        top["nivel"] = label
        resultados[col] = top

        print(f"\n   📌 Top {top_n} por {label}:")
        for _, row in top.iterrows():
            print(f"      {row['categoria']:<35} {row['unidades_vendidas']:>10,} uds  "
                  f"({row['porcentaje']}%)")

    return resultados


# ──────────────────────────────────────────────
# 5. EXPORTACIÓN
# ──────────────────────────────────────────────

def exportar_resultados(
    ventas_diarias: pd.DataFrame,
    top_cats: dict,
    output_dir: str,
) -> None:
    """
    Guarda los archivos procesados en output_dir.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # CSV principal para SARIMA (fecha + ventas)
    path_ventas = os.path.join(output_dir, "hm_ventas_agregadas.csv")
    ventas_diarias[["fecha", "total_articulos"]].to_csv(path_ventas, index=False)
    size_kb = os.path.getsize(path_ventas) / 1024
    print(f"\n💾 Exportado: {path_ventas}  ({size_kb:.1f} KB)")

    # CSV extendido con ingresos (útil para análisis de valor)
    path_ext = os.path.join(output_dir, "hm_ventas_extendidas.csv")
    ventas_diarias.to_csv(path_ext, index=False)
    print(f"💾 Exportado: {path_ext}")

    # CSV de top categorías
    all_tops = pd.concat(top_cats.values(), ignore_index=True)
    path_cats = os.path.join(output_dir, "hm_top_categorias.csv")
    all_tops.to_csv(path_cats, index=False)
    print(f"💾 Exportado: {path_cats}")

    print("\n✅ Todos los archivos exportados correctamente.")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  H&M Dataset — Pipeline de Limpieza y Agregación")
    print("=" * 60)

    # Rutas de archivos (ajusta los nombres si difieren)
    trans_path = os.path.join(args.data_dir, "transactions_train.csv")
    art_path   = os.path.join(args.data_dir, "articles.csv")
    cust_path  = os.path.join(args.data_dir, "customers.csv")

    # Verificar existencia
    for p in [trans_path, art_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"No se encontró: {p}\n"
                f"Asegúrate de que los CSV estén en: {args.data_dir}"
            )

    # Pipeline
    df_trans   = cargar_transacciones(trans_path)
    df_art     = cargar_articulos(art_path)

    if os.path.exists(cust_path):
        df_cust = cargar_clientes(cust_path)  # noqa: F841 — disponible para uso futuro

    df_filtrado    = filtrar_ultimos_meses(df_trans, args.meses)
    ventas_diarias = agregar_ventas_diarias(df_filtrado)
    top_cats       = top_categorias(df_filtrado, df_art, args.top_n)

    exportar_resultados(ventas_diarias, top_cats, args.output_dir)

    print("\n" + "=" * 60)
    print("  ¡Pipeline completado exitosamente!")
    print("  Siguiente paso: cargar hm_ventas_agregadas.csv en SARIMA")
    print("=" * 60)


if __name__ == "__main__":
    main()