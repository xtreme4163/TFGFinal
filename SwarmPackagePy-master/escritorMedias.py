#!/usr/bin/env python3
from pathlib import Path
import re, io, sys
import pandas as pd

ALG_VALIDOS  = {"PSO", "WOA", "GWO", "FA", "ABA"}
FUNC_VALIDOS = {"MSE", "MAE", "SSIM", "UQI"}
HEADERS = ["Imagen","MSE","MAE","SSIM","FSIM","VIF","Tiempo"]

# IQI_algo_func_paleta con extensión opcional (.txt / .TXT / sin)
PATRON  = re.compile(r"^IQI_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d+)(?:\.txt)?$", re.I)

root   = Path(".")       # carpeta donde están los IQI_…
SALIDA = "resumen_IQI.xlsx"

def cargar_valido(fichero: Path):
    """Lee y filtra; muestra cuántas filas quedan."""
    rows = []
    with fichero.open(encoding="utf-8") as fh:
        for raw in fh:
            txt = raw.strip()
            if not txt:
                continue
            if txt.startswith("Ajuste"):
                cod = txt.split(":",1)[1].strip()
                if cod != "0":
                    print(f"    ↳  corto en 'Ajuste: {cod}'")
                    break
                continue
            if txt.startswith("Imagen"):   # cabecera repetida
                continue
            rows.append(raw)

    print(f"    • líneas válidas: {len(rows)}")
    if not rows:
        return None

    return pd.read_csv(io.StringIO("".join(rows)),
                       delim_whitespace=True,
                       names=HEADERS,
                       engine="python")

# ---------- recorrer archivos ----------------------------------------------
hojas = {}
rechazados: set[str] = set()

for f in root.glob("IQI_*"):             
    m = PATRON.match(f.name)
    if not m:
        print(f"✘ Nombre fuera de patrón → {f.name}")
        continue

    alg, func, pal = m.groups()
    pal = int(pal)
    clave = f"{alg}_{func}"
    alg   = alg.upper()
    func  = func.upper()


    if alg not in ALG_VALIDOS or func not in FUNC_VALIDOS:
        # descarta archivos con algoritmo o función no deseados
        rechazados.add(f"{alg}_{func}")
        continue

    print(f"✓ Procesando {f.name}  (paleta {pal})")
    df = cargar_valido(f)
    if df is None:
        print("    ⚠️  sin datos, se omite")
        continue

    resumen = (df.groupby("Imagen", as_index=False)
                 .mean(numeric_only=True)
                 .round(5))
    resumen.insert(0, "N", pal)

    hojas.setdefault(clave, []).append(resumen)

# ---------- grabar Excel si hay algo ---------------------------------------
if not hojas:
    sys.exit("⚠️  No se generó ninguna hoja: revisa los mensajes anteriores.")

with pd.ExcelWriter(SALIDA, engine="openpyxl") as wr:
    for hoja, dfs in hojas.items():
        final = pd.concat(dfs, ignore_index=True).sort_values(["N","Imagen"])
        final.to_excel(wr, sheet_name=hoja[:31], index=False)

print(f"✅ Excel creado: {SALIDA}")

# ---------- informe de combinaciones descartadas ----------
if rechazados:
    print("\nℹ️  Combinaciones Algoritmo_Función descartadas:")
    for combo in sorted(rechazados):
        print("   -", combo)
else:
    print("\nℹ️  No se descartó ninguna combinación fuera de los filtros.")
