import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.signal import convolve2d

# Función para simular la función de degradación (filtro específico)
def degradation_function(image):
    # Filtro de degradación (filtro de paso bajo)
    kernel = np.array([[0.04, 0.08, 0.04],
                       [0.08, 0.16, 0.08],
                       [0.04, 0.08, 0.04]])
    
    # Aplica el filtro de degradación a la imagen
    degraded_image = convolve2d(image, kernel, mode='same', boundary='symm')
    return degraded_image

# Función de filtrado de mínimos cuadrados restringidos para restauración
def constrained_least_squares(image_degraded, kernel, alpha):
    # Calcula la transformada de Fourier del kernel
    kernel_fft = np.fft.fft2(kernel, s=image_degraded.shape)
    
    # Calcula la imagen degradada en el dominio de la frecuencia
    image_degraded_fft = np.fft.fft2(image_degraded)
    
    # Aplica el filtrado de mínimos cuadrados restringidos en el dominio de la frecuencia
    image_restored_fft = np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + alpha) * image_degraded_fft
    
    # Transformada inversa para obtener la imagen restaurada
    image_restored = np.fft.ifft2(image_restored_fft).real
    
    return image_restored

# Ruta de la imagen a cargar (cambia 'ruta/a/tu/imagen.jpg' por la ruta real de tu imagen)
ruta_imagen = 'nn.jpg'

try:
    # Cargar la imagen desde la ruta especificada
    image = io.imread(ruta_imagen)
    
    # Convertir la imagen a escala de grises si es necesario
    if image.ndim == 3:
        image_gray = color.rgb2gray(image)
    else:
        image_gray = image  # La imagen ya está en escala de grises
    
    # Aplicar la función de degradación para obtener la imagen degradada
    image_degraded = degradation_function(image_gray)
    
    # Definir un kernel para el filtrado de mínimos cuadrados restringidos
    # En este caso, un kernel simple de identidad
    kernel = np.eye(3) / 9
    
    # Parámetro de regularización para el filtrado de mínimos cuadrados restringidos
    alpha = 0.01
    
    # Aplicar el filtrado de mínimos cuadrados restringidos para restaurar la imagen
    image_restored = constrained_least_squares(image_degraded, kernel, alpha)
    
    # Mostrar las imágenes original, degradada y restaurada
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_degraded, cmap='gray')
    plt.title('Imagen Degradada')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image_restored, cmap='gray')
    plt.title('Imagen Restaurada (Filtrado de Mínimos Cuadrados Restringidos)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

except IOError:
    print(f"No se pudo cargar la imagen en {ruta_imagen}. Verifica la ruta y el formato de la imagen.")
except Exception as e:
    print(f"Error: {str(e)}")


