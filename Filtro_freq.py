from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np

# # Realizando operação com a biblioteca NUMPY:
# img_original = cv2.imread('relogio.pgm',0) # imagem tons de cinza
# f = np.fft.fft2(img_original) #Calculo da FFT
# fshift_PA = np.fft.fftshift(f) # Desloca a frequência 0 para a origem.
# Espectro_IN = 20*np.log(np.abs(fshift_PA)) #Espectro de magnitude.

# #Aplicação do filtro PASSA-ALTA retangular
# rows,cols = img_original.shape
# crow,ccol = rows/2 , cols/2
# fshift_PA[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 0
# f_ishift_PA = np.fft.ifftshift(fshift_PA)
# img_PA = np.fft.ifft2(f_ishift_PA)
# img_PA = np.abs(img_PA)

# #Realizando operação com a biblioteca NUMPY:
# img_original = cv2.imread('relogio.pgm',0) # imagem tons de cinza
# f = np.fft.fft2(img_original) #Calculo da FFT
# fshift_PB = np.fft.fftshift(f) # Desloca a frequência 0 para a origem.
# Espectro_IN = 20*np.log(np.abs(fshift_PB)) #Espectro de magnitude.


# #Aplicação do filtro PASSA-BAIXA RETANGULAR
# rows,cols = img_original.shape
# crow,ccol = rows/2 , cols/2
# fshift_PB[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 1
# f_ishift_PB = np.fft.ifftshift(fshift_PB)
# img_PB = np.fft.ifft2(f_ishift_PB)
# img_PB = np.abs(img_PB)

# #PLOTS
# plt.subplot(131),plt.imshow(img_original, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(img_PB, cmap = 'gray')
# plt.title('Filtro passa-baixa'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img_PA, cmap = 'gray')
# plt.title('Filtro passa-alta'), plt.xticks([]), plt.yticks([])
# plt.subplot(231),plt.imshow(Espectro_IN, cmap = 'gray')
# plt.title('Espectro original'), plt.xticks([]), plt.yticks([])

# plt.show()


#PRESENÇA DE ARTEFATOS SE DÁ AO UTILIZAR JANELA RETANGULAR. 
#AS MAIS ULTILIZADAS SÃO GAUSSIANAS

#Operando com a biblioteca OPENCV

img = cv2.imread('Lena.pgm',0)
img = np.float32(img)
dft = cv2.dft(img,flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Imagem de entrada'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Espectro original'), plt.xticks([]), plt.yticks([])
plt.show()


# Aplicando Filtros RETANGULARES
# PASSA BAIXA
rows, cols = img.shape
crow,ccol = rows/2 , cols/2

# create a mask first, center square is 1, remaining all zeros
mask_PB = np.zeros((rows,cols,2),np.uint8)
mask_PB[int(crow)-30:int(crow)+30, int(ccol)-30:int(ccol)+30] = 1

# apply mask and inverse DFT
fshift_PB = dft_shift*mask_PB
f_ishift_PB = np.fft.ifftshift(fshift_PB)
img_back_PB = cv2.idft(f_ishift_PB)
img_back_PB = cv2.magnitude(img_back_PB[:,:,0],img_back_PB[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back_PB, cmap = 'gray')
plt.title('Filtro passa baixa'), plt.xticks([]), plt.yticks([])

plt.show()

#FILTRO PASSA ALTA

# apply mask and inverse DFT
fshift_PA = dft_shift*(1-mask_PB)
f_ishift_PA = np.fft.ifftshift(fshift_PA)
img_back_PA = cv2.idft(f_ishift_PA)
img_back_PA = cv2.magnitude(img_back_PA[:,:,0],img_back_PA[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Imagem de entrada'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back_PA, cmap = 'gray')
plt.title('Filtro passa alta'), plt.xticks([]), plt.yticks([])

plt.show()


#Filtros circulares

# Circular LPF mask, center circle is 1, remaining all zeros
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols, 2), np.uint8)
r = 70
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

# apply mask and inverse DFT
fshift = dft_shift * mask

fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('After FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
plt.show()


# Circular HPF mask, center circle is 0, remaining all zeros

# apply mask and inverse DFT
fshift = dft_shift * (1-mask)

fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('After FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
plt.show()




#CIRCULAR PASSA BANDA
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)  # center

# Concentric BPF mask,with are between the two cerciles as one's, rest all zero's.
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols, 2), np.uint8)
r_out = 80
r_in = 5
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]

mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
mask[mask_area] = 1

# apply mask and inverse DFT
fshift = dft_shift * mask

fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('After FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
plt.show()