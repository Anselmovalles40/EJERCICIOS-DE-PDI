import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar una imagen de ejemplo
image = cv2.imread('cabra.jpg', cv2.IMREAD_GRAYSCALE)

# Crear un kernel de desenfoque por movimiento
size = 15
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# Aplicar el desenfoque por movimiento a la imagen
motion_blur_image = cv2.filter2D(image, -1, kernel_motion_blur)

# Restaurar la imagen utilizando la deconvolución de Wiener (simple)
def wiener_filter(img, kernel, K):
    kernel = np.fft.fft2(kernel, s=img.shape)
    img = np.fft.fft2(img)
    kernel_conj = np.conj(kernel)
    deconvolved = (img * kernel_conj) / (kernel * kernel_conj + K)
    deconvolved = np.abs(np.fft.ifft2(deconvolved))
    return deconvolved

restored_image = wiener_filter(motion_blur_image, kernel_motion_blur, 0.01)

# Mostrar las imágenes
titles = ['Imagen Original', 'Imagen con Desenfoque', 'Imagen Restaurada']
images = [image, motion_blur_image, restored_image]

plt.figure(figsize=(10, 8))
for i in range(len(images)):
    plt.subplot(1, 3, i + 1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
