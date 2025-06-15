from pathlib import Path, PurePath
import itertools, textwrap

ROOT   = Path(".")          # ajusta si tus ficheros están en otra carpeta
muestra = 3                 # cuántas líneas mostrar de cada archivo

files = sorted(ROOT.glob("IQI*"))
print(f"\nArchivos que empiezan por 'IQI' encontrados: {len(files)}\n")

for f in files[:20]:        # lista los 20 primeros como máximo
    print(f"─ {f.name}")
    try:
        with f.open(encoding="utf-8") as h:
            for i, line in zip(range(muestra), h):
                print("   ", textwrap.shorten(line.rstrip(), 120))
    except Exception as e:
        print("   [Error leyendo]", e)
    print()
