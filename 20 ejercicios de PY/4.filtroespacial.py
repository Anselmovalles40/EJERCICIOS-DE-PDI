import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar una imagen de ejemplo
image = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

# Añadir ruido gaussiano a la imagen
noise_sigma = 25
noisy_image = image + noise_sigma * np.random.randn(*image.shape)
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Aplicar filtro promedio
average_filter = cv2.blur(noisy_image, (3, 3))

# Aplicar filtro de mediana
median_filter = cv2.medianBlur(noisy_image, 3)

# Aplicar filtro gaussiano
gaussian_filter = cv2.GaussianBlur(noisy_image, (3, 3), 0)

# Mostrar las imágenes
titles = ['Imagen Original', 'Imagen con Ruido', 'Filtro Promedio', 'Filtro de Mediana', 'Filtro Gaussiano']
images = [image, noisy_image, average_filter, median_filter, gaussian_filter]

plt.figure(figsize=(10, 8))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
