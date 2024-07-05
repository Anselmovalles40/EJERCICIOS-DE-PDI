import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import color, restoration, io
from skimage.util import random_noise

# Ruta de tu imagen (cambia 'ruta/a/tu/imagen.jpg' por la ruta real de tu imagen)
ruta_imagen = 'animo.jpg'

try:
    # Cargar la imagen y convertir a escala de grises
    image = io.imread(ruta_imagen, as_gray=True)

    # Crear una PSF de desenfoque por movimiento
    psf = np.zeros((15, 15))
    psf[7, :] = 1
    psf = psf / psf.sum()

    # Degradar la imagen con la PSF y añadir ruido gaussiano
    degraded = convolve2d(image, psf, 'same')
    degraded = random_noise(degraded, mode='gaussian', var=0.01)

    # Estimar la PSF y la imagen original utilizando deconvolución ciega
    deconvolved, _ = restoration.unsupervised_wiener(degraded, psf)

    # Mostrar las imágenes
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(degraded, cmap='gray')
    plt.title('Imagen Degradada')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(deconvolved, cmap='gray')
    plt.title('Imagen Restaurada')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(psf, cmap='hot')
    plt.title('PSF Estimada')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except IOError:
    print(f"No se pudo cargar la imagen en {ruta_imagen}. Verifica la ruta y el formato de la imagen.")
except Exception as e:
    print(f"Error: {str(e)}")



