import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, data
from skimage.transform import radon, iradon_sart, iradon

# Función para cargar y mostrar la imagen
def load_and_show_image(image_path):
    try:
        # Cargar la imagen desde la ruta especificada
        image = io.imread(image_path, as_gray=True)
        
        # Mostrar la imagen original
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title('Imagen Original')
        plt.axis('off')
        plt.show()
        
        # Definir ángulos de proyección
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        
        # Generar proyecciones utilizando la transformada de Radón
        sinogram = radon(image, theta=theta, circle=True)
        
        # Reconstruir la imagen a partir del sinograma utilizando diferentes métodos
        # Método de reconstrucción usando SART (Simultaneous Algebraic Reconstruction Technique)
        reconstructed_image_sart = iradon_sart(sinogram, theta=theta)
        
        # Método de reconstrucción usando la transformada de Radón inversa estándar
        reconstructed_image_standard = iradon(sinogram, theta=theta, circle=True)
        
        # Mostrar las imágenes reconstruidas
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(reconstructed_image_sart, cmap='gray')
        plt.title('Reconstrucción con SART')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image_standard, cmap='gray')
        plt.title('Reconstrucción Estándar')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except IOError:
        print(f"No se pudo cargar la imagen en {image_path}. Verifica la ruta y el formato de la imagen.")
    except Exception as e:
        print(f"Error: {str(e)}")

# Ruta de la imagen a cargar (cambia 'ruta/a/tu/imagen.jpg' por la ruta real de tu imagen)
ruta_imagen = 'ff.jpg'

# Cargar y mostrar la imagen, generar proyecciones y reconstruir la imagen
load_and_show_image(ruta_imagen)
