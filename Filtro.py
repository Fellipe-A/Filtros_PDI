from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np

# Leitura da imagem
img_original = cv2.imread('Lena.pgm',0) # imagem tons de cinza
img_original = img_original.astype(float)

Filtro = input("""Selecione o filtro de entrada:\n
 1 = Filtro da média;\n 2 = Filtro da mediana;\n 3 = Filtro Gaussiano;
 4 = Laplaciano;\n 5 = Filtro de Sobel\n 6 =Filtro bilateral
 7 = Filtro minimo;\n 8 = Filtro maximo
""")
Filtro_INT  = int(Filtro)

if Filtro_INT == 1:
#Aplicação de filtro da média
    Tamanho = input('Informe o tamanho da máscara\n')
    Tamanho_INT = int(Tamanho)
    kernel = np.ones((Tamanho_INT,Tamanho_INT),np.float32)/Tamanho_INT*Tamanho_INT
    img_Filtrada = cv2.filter2D(img_original,-1,kernel)

elif Filtro_INT == 2:
#Aplicação de filtro da mediana
    Tamanho = input('Informe o tamanho da máscara (k = 3 || k = 5) \n')
    Tamanho_INT = int(Tamanho)
    img_original = img_original.astype('float32')
    img_Filtrada = cv2.medianBlur(img_original,Tamanho_INT)

elif Filtro_INT == 3:
#Aplicação de filtro Gaussiano
    Tamanho = input('Informe o tamanho da máscara (APENAS VALOR IMPAR)\n')
    Tamanho_INT = int(Tamanho)
    img_Filtrada = cv2.GaussianBlur(img_original,(Tamanho_INT,Tamanho_INT),0,-1,0,cv2.BORDER_DEFAULT)
    #OPÇÕES PARA BORDAS:
    # BORDER_CONSTANT
    # BORDER_REPLICATE
    # BORDER_REFLECT
    # BORDER_WRAP
    # BORDER_REFLECT
    # BORDER_TRANSPARENT
    # BORDER_REFLECT101
    # BORDER_ISOLATED

elif Filtro_INT == 4:
    #Aplicação do filtro Laplaciano
    Laplaciano = cv2.Laplacian(img_original,cv2.CV_64F)    
    plt.imshow(Laplaciano)
    img_Filtrada = img_original + Laplaciano

    plt.subplot(1,3,1)
    plt.imshow(img_original,cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(Laplaciano,cmap='gray')
    plt.axis('off')


    plt.subplot(1,3,3)
    plt.imshow(img_Filtrada,cmap='gray')
    plt.axis('off')

    plt.show()
    exit()

elif Filtro_INT == 5:
    #Aplicação do filtro de Sobel
    img_Filtrada = cv2.Sobel(img_original,-1,2,2) #Calculando derivadas de segunda ordem!

elif Filtro_INT == 6:
    #Aplicação do filtro bilateral
    img_original = img_original.astype('float32')
    #Parâmetros: 10 - largura do filtro, 100 - Sigma color (>150)
    img_Filtrada = cv2.bilateralFilter(img_original,10,150,150)

elif Filtro_INT == 7:
#Aplicação de filtro da média
    Tamanho = input('Informe o tamanho da máscara\n')
    Tamanho_INT = int(Tamanho)
    kernel = np.ones((Tamanho_INT,Tamanho_INT),np.float32)/Tamanho_INT*Tamanho_INT
    img_Filtrada = cv2.filter2D(img_original,-1,kernel)

elif Filtro_INT == 8:
#Aplicação de filtro da média
    Tamanho = input('Informe o tamanho da máscara\n')
    Tamanho_INT = int(Tamanho)
    kernel = np.ones((Tamanho_INT,Tamanho_INT),np.float32)/Tamanho_INT*Tamanho_INT
    img_Filtrada = cv2.filter2D(img_original,-1,kernel)

else: 
    print('Escolha um filtro válido')
    exit()

#Plot
plt.subplot(1,2,1)
plt.imshow(img_original,cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img_Filtrada,cmap='gray')
plt.axis('off')

plt.show()