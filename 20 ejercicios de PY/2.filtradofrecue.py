import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para aplicar la Transformada de Fourier y centrar la imagen
def fft_image(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return dft_shift, magnitude_spectrum

# Función para aplicar la Transformada Inversa de Fourier
def ifft_image(dft_shift):
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back

# Cargar una imagen de ejemplo y añadir ruido periódico
image = cv2.imread('Kanye.jpg', cv2.IMREAD_GRAYSCALE)

# Añadir ruido periódico (para demostración)
rows, cols = image.shape
sinusoidal_noise = np.sin(2 * np.pi * 10 * np.arange(cols) / cols) * 128 + 128
noisy_image = image + sinusoidal_noise

# Aplicar la Transformada de Fourier a la imagen ruidosa
dft_shift, magnitude_spectrum = fft_image(noisy_image)

# Crear un filtro de rechazo de bandas para eliminar el ruido periódico
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-5:crow+5, ccol-30:ccol+30] = 0
mask[crow-5:crow+5, ccol-5:ccol+5] = 1  # Preserva el componente DC (frecuencia cero)

# Aplicar el filtro
dft_shift_filtered = dft_shift * mask

# Aplicar la Transformada Inversa de Fourier
restored_image = ifft_image(dft_shift_filtered)

# Mostrar las imágenes
titles = ['Imagen Original', 'Imagen con Ruido', 'Espectro de Magnitud', 'Imagen Restaurada']
images = [image, noisy_image, magnitude_spectrum, restored_image]

plt.figure(figsize=(10, 8))
for i in range(len(images)):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
