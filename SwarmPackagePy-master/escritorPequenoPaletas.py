#!/usr/bin/env python3
"""
Produce: resumen_global.xlsx  (hoja única “Datos”)
Una fila = una paleta (32·64·128·256) de cada combinación ALG + FUN
Valores = media y σ globales sobre TODAS las imágenes.
"""

from pathlib import Path
import re, io, sys
import pandas as pd

# --- parámetros editables ---------------------------------------------------
ROOT     = Path(".")                           # carpeta con los IQI_*.txt
SALIDA   = "resumen_global.xlsx"

ALG_OK   = {"PSO","WOA","GWO","FA","ABA"}
FUNC_OK  = {"MSE","MAE","SSIM","UQI"}
METRICAS = ["MSE","MAE","SSIM","FSIM","VIF"]   # tiempo va aparte
PATRON   = re.compile(r"^IQI_([^_]+)_([^_]+)_(\d+)(?:\.txt)?$", re.I)
# ----------------------------------------------------------------------------

def leer_txt(path: Path):
    """Devuelve DataFrame con columnas Imagen + métricas + Tiempo (float)."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            t = raw.strip()
            if not t:
                continue
            if t.lower().startswith("ajuste"):
                if ":" in t and t.split(":",1)[1].strip() == "1":
                    break          # ‘Ajuste: 1’  → descarta resto del archivo
                continue
            if t.lower().startswith("imagen"):
                continue           # omite cabecera repetida
            rows.append(raw)

    if not rows:
        return None

    df = pd.read_csv(
        io.StringIO("".join(rows)),
        sep=r"\s+", header=None, dtype=str, engine="python"
    )
    if df.shape[1] < 7:
        return None

    df = df.iloc[:, :7]
    df.columns = ["Imagen"] + METRICAS + ["Tiempo"]

    # coma → punto  y  a float
    num_cols = METRICAS + ["Tiempo"]
    df[num_cols] = (df[num_cols]
                    .apply(lambda col: col.str.replace(",", ".", regex=False))
                    .apply(pd.to_numeric, errors="coerce"))
    return df

# -------- leer todos los TXT y apilar ---------------------------------------
todo = []   # columnas: ALG FUN COL métrica1 … Tiempo
for txt in ROOT.glob("IQI_*"):
    m = PATRON.match(txt.name)
    if not m:
        continue
    alg, func, pal = map(str.upper, m.groups())
    if alg not in ALG_OK or func not in FUNC_OK:
        continue
    pal = int(pal)

    df = leer_txt(txt)
    if df is None:
        continue

    todo.append(df.assign(ALG=alg, FUN=func, COL=pal))

if not todo:
    sys.exit("⚠️  No se encontraron TXT válidos.")

datos = pd.concat(todo, ignore_index=True)

# -------- agregación global: media y σ sobre todas las imágenes -------------
aggs = {}
for met in METRICAS + ["Tiempo"]:
    aggs[f"{met}_x̅"] = (met, "mean")
    aggs[f"{met}_σ"]  = (met, "std")

resumen = (datos
           .groupby(["ALG","FUN","COL"])
           .agg(**aggs)
           .round({"MSE_x̅":2, "MSE_σ":2,
                   "MAE_x̅":2, "MAE_σ":2,
                   "SSIM_x̅":4, "SSIM_σ":4,
                   "FSIM_x̅":4, "FSIM_σ":4,
                   "VIF_x̅":4, "VIF_σ":4,
                   "Tiempo_x̅":2, "Tiempo_σ":2})
           .reset_index()
           .sort_values(["ALG","FUN","COL"])
)

# -------- exportar ----------------------------------------------------------
with pd.ExcelWriter(SALIDA, engine="openpyxl") as wr:
    resumen.to_excel(wr, sheet_name="Datos", index=False)

print(f"✅  Libro generado: {SALIDA}")
