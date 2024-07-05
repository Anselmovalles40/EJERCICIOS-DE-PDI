import numpy as np
import matplotlib.pyplot as plt
from skimage import io, data, transform
from skimage.transform import radon, iradon

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
        
        # Generar proyecciones (simulación de datos de tomografía)
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        sinogram = radon(image, theta=theta, circle=True)
        
        # Reconstruir la imagen a partir de las proyecciones
        reconstructed_image = iradon(sinogram, theta=theta, circle=True)
        
        # Mostrar la imagen reconstruida
        plt.figure(figsize=(6, 6))
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title('Imagen Reconstruida')
        plt.axis('off')
        plt.show()
        
    except IOError:
        print(f"No se pudo cargar la imagen en {image_path}. Verifica la ruta y el formato de la imagen.")
    except Exception as e:
        print(f"Error: {str(e)}")

# Ruta de la imagen a cargar (cambia 'ruta/a/tu/imagen.jpg' por la ruta real de tu imagen)
ruta_imagen = 'f.jpg'

# Cargar y mostrar la imagen, generar proyecciones y reconstruir la imagen
load_and_show_image(ruta_imagen)
