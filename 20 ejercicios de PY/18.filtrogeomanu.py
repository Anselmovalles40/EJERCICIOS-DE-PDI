import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# Función para cargar y mostrar la imagen
def load_and_show_image(image_path):
    try:
        # Cargar la imagen desde la ruta especificada
        image = io.imread(image_path)
        
        # Convertir la imagen a escala de grises si es necesario
        if image.ndim == 3:
            image_gray = color.rgb2gray(image)
        else:
            image_gray = image  # La imagen ya está en escala de grises
        
        # Aplicar el filtro medio geométrico
        filtered_image = geometric_mean_filter(image_gray, size=3)
        
        # Mostrar las imágenes original y filtrada
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image_gray, cmap='gray')
        plt.title('Imagen Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(filtered_image, cmap='gray')
        plt.title('Imagen Filtrada (Filtro Medio Geométrico)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except IOError:
        print(f"No se pudo cargar la imagen en {image_path}. Verifica la ruta y el formato de la imagen.")
    except Exception as e:
        print(f"Error: {str(e)}")

# Función para aplicar el filtro medio geométrico
def geometric_mean_filter(image, size):
    # Dimensiones de la imagen
    rows, cols = image.shape
    
    # Tamaño del filtro
    pad = size // 2
    
    # Crear una matriz para almacenar la imagen filtrada
    filtered_image = np.zeros_like(image)
    
    # Recorrer la imagen y aplicar el filtro medio geométrico
    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            # Obtener la submatriz correspondiente al filtro
            window = image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            
            # Calcular la media geométrica
            geometric_mean = np.prod(window) ** (1.0 / (size * size))
            
            # Asignar el valor de la media geométrica a la imagen filtrada
            filtered_image[i, j] = geometric_mean
    
    return filtered_image

# Ruta de la imagen a cargar (cambia 'ruta/a/tu/imagen.jpg' por la ruta real de tu imagen)
ruta_imagen = 'u.jpg'

# Cargar y mostrar la imagen con el filtro aplicado
load_and_show_image(ruta_imagen)
