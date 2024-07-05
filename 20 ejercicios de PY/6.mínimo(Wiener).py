import numpy as np
import matplotlib.pyplot as plt
from skimage import color, restoration, io
from skimage.util import random_noise
from scipy.signal import convolve2d

# Ruta de tu imagen (cambia 'ruta/a/tu/imagen.jpg' por la ruta real de tu imagen)
ruta_imagen = '777.jpg'

try:
    # Cargar la imagen en escala de grises
    image = io.imread(ruta_imagen, as_gray=True)

    # Simular la degradación de la imagen
    psf = np.ones((5, 5)) / 25  # Kernel de filtro de desenfoque (filtro promedio)
    degraded = convolve2d(image, psf, mode='same', boundary='symm')
    degraded = random_noise(degraded, mode='gaussian', var=0.01)

    # Restaurar la imagen utilizando el filtro de Wiener
    restored = restoration.wiener(degraded, psf, 0.1)

    # Mostrar las imágenes
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(degraded, cmap='gray')
    plt.title('Imagen Degradada')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(restored, cmap='gray')
    plt.title('Imagen Restaurada (Filtro de Wiener)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except IOError:
    print(f"No se pudo cargar la imagen en {ruta_imagen}. Verifica la ruta y el formato de la imagen.")
except Exception as e:
    print(f"Error: {str(e)}")
