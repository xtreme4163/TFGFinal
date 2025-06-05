#!/usr/bin/env python3
from pathlib import Path
import re, io, sys
import pandas as pd

ROOT   = Path(".")
OUT_XL = "resumen_IQI.xlsx"

ALG_OK   = {"PSO","WOA","GWO","FA","ABA"}
FUNC_OK  = {"MSE","MAE","SSIM","UQI"}
METRICAS = ["MSE","MAE","SSIM","FSIM","VIF","Tiempo"]
HEADERS  = ["Imagen"] + METRICAS
PATRON   = re.compile(r"^IQI_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d+)(?:\.txt)?$", re.I)

# ---------- funciones -------------------------------------------------------
def leer(path: Path):
    """
    Devuelve DataFrame con los datos válidos del archivo.
    • Ignora cualquier línea que empiece por 'Ajuste'.
    • Si es 'Ajuste: <cod>' y <cod> != 0  →  corta la lectura.
    • Ignora cabeceras repetidas y líneas con nº de columnas distinto de 7.
    """
    rows = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            t = raw.lstrip()                 # quitar espacios al inicio
            if not t.strip():
                continue                     # línea vacía

            if t.lower().startswith("ajuste"):   # 'ajuste' o 'Ajuste'
                # ¿Hay código detrás del ':'?
                if ":" in t:
                    cod = t.split(":", 1)[1].strip()
                    if cod and cod != "0":   # Ajuste distinto de 0 → cortamos
                        break
                # Sea 'Ajuste:' o 'Ajuste: 0' → la saltamos
                continue

            if t.lower().startswith("imagen"):
                continue                     # cabecera repetida

            # Asegurarse de que hay 7 columnas
            if len(t.split()) != 7:
                continue                     # descarta líneas mal formadas

            rows.append(raw)

    if not rows:
        return None

    return pd.read_csv(
        io.StringIO("".join(rows)),
        sep=r"\s+",               # igual que antes, pero sin 'delim_whitespace'
        names=HEADERS,
        engine="python"
    )


def aplanar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                  for c in df.columns]
    return df

def redondear(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica 2 o 4 decimales según la métrica."""
    for col in df.columns:
        if col in ("N", "Imagen"):
            continue
        base = col.split("_")[0]     # MSE_media → MSE
        if base in {"MSE","MAE"}:
            df[col] = df[col].round(2)
        elif base in {"SSIM","FSIM","VIF"}:
            df[col] = df[col].round(4)
        elif base == "Tiempo":
            df[col] = df[col].round(2)   
    return df

# ---------- procesamiento ---------------------------------------------------
hojas, omitidos = {}, set()

for f in ROOT.glob("IQI_*"):
    m = PATRON.match(f.name)
    if not m:
        continue
    alg, func, pal = map(str.upper, m.groups())
    if alg not in ALG_OK or func not in FUNC_OK:
        omitidos.add(f"{alg}_{func}"); continue

    df = leer(f)
    if df is None:
        continue
    pal = int(pal)

    stats = (df.groupby("Imagen")
               .agg(['mean','std']))

    # Renombra ('mean'→'media', 'std'→'sd')
    stats.columns = [(m,'media' if s=='mean' else 'sd') for m,s in stats.columns]

    stats.insert(0, ('N',''), pal)
    stats.reset_index(inplace=True)

    plano = aplanar(stats)
    plano = redondear(plano)

    hojas.setdefault(f"{alg}_{func}", []).append(plano)

# ---------- exportar --------------------------------------------------------
if not hojas:
    sys.exit("⚠️  Ningún archivo válido tras los filtros.")

with pd.ExcelWriter(OUT_XL, engine="openpyxl") as wr:
    for nombre, dfs in hojas.items():
        hoja = pd.concat(dfs, ignore_index=True).sort_values(["N","Imagen"])
        hoja.to_excel(wr, sheet_name=nombre[:31], index=False)

print(f"✅ Excel creado: {OUT_XL}")

if omitidos:
    print("\nℹ️  Algoritmo_Función descartados:")
    for c in sorted(omitidos):
        print("   -", c)
else:
    print("\nℹ️  No se descartó ninguna combinación.")
