import numpy as np
import matplotlib.pyplot as plt
from skimage import io, restoration
from scipy.signal import convolve2d
from skimage.util import random_noise

# Ruta de tu imagen (cambia 'ruta/a/tu/imagen.jpg' por la ruta real de tu imagen)
ruta_imagen = 'hitemup.jpg'

try:
    # Cargar la imagen en escala de grises
    image = io.imread(ruta_imagen, as_gray=True)

    # Paso 1: Degradación de la imagen
    # Simular desenfoque usando un filtro promedio (PSF - Point Spread Function)
    psf = np.ones((5, 5)) / 25
    image_blurred = convolve2d(image, psf, mode='same', boundary='symm')

    # Añadir ruido gaussiano a la imagen desenfocada
    image_noisy = random_noise(image_blurred, mode='gaussian', var=0.01)

    # Paso 2: Restauración de la imagen
    # Utilizar deconvolución de Richardson-Lucy para restaurar la imagen degradada
    num_iter = 30  # Número de iteraciones para la deconvolución
    restored_image = restoration.richardson_lucy(image_noisy, psf, num_iter=num_iter)

    # Mostrar las imágenes
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_noisy, cmap='gray')
    plt.title('Imagen Degradada (Desenfoque + Ruido)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(restored_image, cmap='gray')
    plt.title('Imagen Restaurada (Richardson-Lucy)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except IOError:
    print(f"No se pudo cargar la imagen en {ruta_imagen}. Verifica la ruta y el formato de la imagen.")
except Exception as e:
    print(f"Error: {str(e)}")