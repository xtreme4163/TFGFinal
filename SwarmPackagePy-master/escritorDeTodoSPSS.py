#!/usr/bin/env python3
"""
Convierte un libro con hojas <ALG>_<FUNC> en uno solo con la hoja “Datos”.
Solo se conservan las columnas de *media*:

    ALG_FFUNC_E<METRICA>

Las columnas …_sd se ignoran.
"""

# ─── AJUSTA SOLO ESTA RUTA ─────────────────────────────────────
ORIGINAL_EXCEL = "resumenGrandeMediaDesviacion.xlsx"
# ───────────────────────────────────────────────────────────────

import re, sys
from pathlib import Path
import pandas as pd

IN_FILE  = Path(ORIGINAL_EXCEL)
if not IN_FILE.exists():
    sys.exit(f"❌  No se encontró {IN_FILE.absolute()}")

OUT_FILE = IN_FILE.with_stem(IN_FILE.stem + "_SPSS")

METRICAS = ["MSE", "MAE", "SSIM", "FSIM", "VIF", "Tiempo"]
PAT_HOJA = re.compile(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)$")   # PSO_MSE …

xls = pd.ExcelFile(IN_FILE)
tablas = []

for hoja in xls.sheet_names:
    m = PAT_HOJA.match(hoja)
    if not m:
        continue

    alg, func = m.groups()
    df = pd.read_excel(xls, sheet_name=hoja)

    # ----- renombrar SOLO columnas de media --------------------
    ren = {}
    for met in METRICAS:
        col_media = next(
            (c for c in df.columns
             if c.lower().startswith(met.lower())
             and ("media" in c.lower() or c.lower() == met.lower())),
            None
        )
        if col_media:
            ren[col_media] = f"{alg}_F{func}_E{met}"

    df = df.rename(columns=ren)

    # descarta cualquier columna no renombrada (incluidas las _sd)
    df = df[["N", "Imagen"] + list(ren.values())]

    tablas.append(df)

if not tablas:
    sys.exit("❌  No se encontró ninguna hoja válida ALG_FUNC.")

# ── fusionar sobre claves N + Imagen, combinando duplicados ──
def fusionar(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(b, on=["N", "Imagen"], how="outer", suffixes=("", "_dup"))
    for c in [c for c in m.columns if c.endswith("_dup")]:
        base = c[:-4]
        m[base] = m[base].combine_first(m[c])
        m.drop(columns=c, inplace=True)
    return m

datos = tablas[0]
for t in tablas[1:]:
    datos = fusionar(datos, t)

# ordenar columnas: primero N, Imagen, luego alfabético
fijas = ["N", "Imagen"]
datos = datos[fijas + sorted([c for c in datos.columns if c not in fijas])]

# ── exportar ──────────────────────────────────────────────────
with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as wr:
    datos.to_excel(wr, sheet_name="Datos", index=False)

print(f"✅  Libro generado sin columnas _sd → {OUT_FILE.name}")
