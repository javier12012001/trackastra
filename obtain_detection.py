import numpy as np
import tifffile as tiff
from scipy.ndimage import center_of_mass
import sys

# --- RUTA DE CARGA ---
gt_track_pila_path = r'datasets_externos\Training datasets\Fluo-N2DL-HeLa\01_GT\TRA\man_track000.tif'

try:
    # Cargar el TIFF como una lista de páginas
    tif = tiff.TiffFile(gt_track_pila_path)
    pages = tif.pages

    # Cada página es un frame, pero viene como un vector 1D → reconstruimos
    frames = []
    for p in pages:
        arr = p.asarray()

        
        if arr.ndim == 2:
            frames.append(arr)
        else:
            
            EXPECTED_SHAPE = (512, 512)
            if arr.size == EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]:
                arr = arr.reshape(EXPECTED_SHAPE)
            else:
                raise ValueError(f"El frame tiene tamaño inesperado {arr.size}. No coincide con 512×512.")

            frames.append(arr)

    gt_pila = np.stack(frames, axis=0)
    print(f" GT cargado correctamente. Forma final: {gt_pila.shape}")

    # Buscar primer frame con etiquetas
    first_valid_frame_index = -1
    for t in range(min(len(gt_pila), 20)):
        if np.max(gt_pila[t]) > 0:
            first_valid_frame_index = t
            break

    if first_valid_frame_index == -1:
        print("\nERROR: No se encontraron células en los primeros 20 frames.")
        sys.exit(1)

    print(f" Primer frame válido: t = {first_valid_frame_index}")

    gt_labels = gt_pila[first_valid_frame_index]

except Exception as e:
    print(f"\n ERROR al cargar el TIFF: {e}")
    sys.exit(1)


# --- 4. EXTRACCIÓN DE DETECCIONES (p_i y z_i) ---

cell_ids = np.unique(gt_labels)
cell_ids = cell_ids[cell_ids != 0]  # quitar fondo

centers_p = []
features_z = []

for cell_id in cell_ids:
    mask = (gt_labels == cell_id)

    (cy, cx) = center_of_mass(mask)
    area = mask.sum()

    centers_p.append([cy, cx])
    features_z.append([area])

centers_p = np.array(centers_p)
features_z = np.array(features_z)

print("\n--- Resultados de la Extracción de Detecciones ---")
print(f"Total de detecciones (di): {centers_p.shape[0]}")
print(f"Forma de centros (pi): {centers_p.shape}")
print(f"Forma de características (zi): {features_z.shape}")
