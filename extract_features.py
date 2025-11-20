import tifffile as tiff
import matplotlib.pyplot as plt
import os

# Ruta de ejemplo para una secuencia de imágenes (Stage 1)
data_path = 'datasets_externos\\Training datasets\\Fluo-N2DL-HeLa\\01\\t001.tif' 
img_path = 'datasets_externos\\Training datasets\\Fluo-N2DL-HeLa\\01_GT\\TRA\\man_track001.tif' 

# Cargar una imagen (frame inicial)
try:
    # Cargar la imagen de entrada (intensidad)
    img = tiff.imread(img_path)
    # Cargar la etiqueta de seguimiento (Ground Truth, GT)
    gt_track = tiff.imread(data_path)

    print(f"\n Imagen de entrada cargada. Forma (tamaño): {img.shape}, Tipo: {img.dtype}")
    print(f" Etiqueta de Seguimiento (GT) cargada. Forma (tamaño): {gt_track.shape}, Tipo: {gt_track.dtype}")
    
    # Mostrar la imagen como prueba visual de la carga
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Frame t=000 (01)')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # Los valores del ground truth representan IDs de células
    plt.imshow(gt_track, cmap='nipy_spectral') 
    plt.title('Ground Truth ')
    plt.axis('off')
    
    plt.show()
    
except FileNotFoundError:
    print(f"\n Error: No se encontraron los archivos TIF en las rutas esperadas.")
    print("Asegúrese de que el archivo ZIP se descomprimió correctamente.")
except Exception as e:
    print(f"\n Error al cargar/mostrar archivos TIF: {e}")