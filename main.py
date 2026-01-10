from low_poly import *
from quantize import *
from utils import *

# --- CONFIGURAZIONE ---
FILE_PATH = "modello_convertito.obj"
OUTPUT_IMG = "render_contrast.jpg"

# METODO 1: Quadric Decimation (raccomandato)
METODO = None
# Percentuale di RIDUZIONE: 0.5 = togli 50% delle facce
# 0.1 = togli 10%, 0.9 = togli 90% (molto low-poly)
REDUCTION_PERCENT = 0.1  # Riduci del 50%
# METODO 2: Voxel-based (alternativo)
# Più alto = più low-poly (prova 0.01 - 0.1)
VOXEL_SIZE = 0.1
# Scegli il metodo: 'quadric' o 'voxel' o 'none' (per non semplificare)

def main():
    mesh = get_mesh(FILE_PATH)
    mesh.show()

if __name__ == "__main__":
    main()