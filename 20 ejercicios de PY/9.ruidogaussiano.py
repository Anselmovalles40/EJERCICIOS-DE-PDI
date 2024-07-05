import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util

# Ruta de tu imagen (cambia 'ruta/a/tu/imagen.jpg' por la ruta real de tu imagen)
ruta_imagen = 'loser.jpg'

try:
    # Cargar la imagen
    image = io.imread(ruta_imagen, as_gray=True)

    # Añadir ruido gaussiano a la imagen
    var = 0.01  # Varianza del ruido gaussiano
    image_noisy = util.random_noise(image, mode='gaussian', var=var)

    # Mostrar la imagen original y la imagen con ruido
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_noisy, cmap='gray')
    plt.title(f'Imagen con Ruido Gaussiano\n(varianza={var})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except IOError:
    print(f"No se pudo cargar la imagen en {ruta_imagen}. Verifica la ruta y el formato de la imagen.")
except Exception as e:
    print(f"Error: {str(e)}")