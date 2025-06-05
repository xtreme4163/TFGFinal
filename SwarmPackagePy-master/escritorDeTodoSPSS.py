#!/usr/bin/env python3
"""
Lee un libro Excel con varias hojas <ALG>_<FUNC> y crea un nuevo libro
con una única hoja "Datos" en la que cada fila (N, Imagen) contiene
todas las métricas de todos los algoritmos y funciones:

    ALG_FFUNC_EMETRICA         (media)
    ALG_FFUNC_EMETRICA_sd      (desviación típica)
"""

# ─── AJUSTA AQUÍ SOLO ESTA VARIABLE ─────────────────────────────
ORIGINAL_EXCEL = "resumenGrandeMediaDesviacion.xlsx"   # <-- tu archivo
# ─────────────────────────────────────────────────────────────────

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
        print(f"· Omito hoja '{hoja}'")
        continue

    alg, func = m.groups()
    df = pd.read_excel(xls, sheet_name=hoja)

    # Renombrar columnas locales a la forma ALG_FFUNC_E<métrica>...
    ren = {}
    for met in METRICAS:
        base = met.lower()
        col_media = next((c for c in df.columns
                          if c.lower().startswith(base) and
                             ("media" in c.lower() or c.lower() == met)), None)
        col_sd = next((c for c in df.columns
                       if c.lower().startswith(base) and "sd" in c.lower()), None)

        if col_media:
            ren[col_media] = f"{alg}_F{func}_E{met}"
        if col_sd:
            ren[col_sd]    = f"{alg}_F{func}_E{met}_sd"

    df = df.rename(columns=ren)
    tablas.append(df)

if not tablas:
    sys.exit("❌  No hay hojas válidas ALG_FUNC en el libro.")

# ---------------------------------------------------------------
# Fusionar UNA sola matriz sobre las claves N + Imagen
# ---------------------------------------------------------------
def fusionar(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Merge externo y combina columnas duplicadas (_dup)."""
    merged = left.merge(right, on=["N", "Imagen"], how="outer",
                        suffixes=("", "_dup"))

    # para cada columna *_dup, rellena la principal y elimina la copia
    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    for c_dup in dup_cols:
        c = c_dup[:-4]          # nombre sin sufijo
        merged[c] = merged[c].combine_first(merged[c_dup])
        merged.drop(columns=c_dup, inplace=True)
    return merged

datos = tablas[0]
for t in tablas[1:]:
    datos = fusionar(datos, t)

# Opcional: ordena columnas dejando N e Imagen delante
fijas = ["N", "Imagen"]
resto = sorted([c for c in datos.columns if c not in fijas])
datos = datos[fijas + resto]

# ---------------------------------------------------------------
# Exportar
# ---------------------------------------------------------------
with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as wr:
    datos.to_excel(wr, sheet_name="Datos", index=False)

print(f"✅  Libro generado correctamente: {OUT_FILE.name}")
