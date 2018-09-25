#from svd_power import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from svd_power import *


#Find SVD for Matrix A
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
	
img = mpimg.imread('vault_boy.gif')     
gray = rgb2gray(img)


x = np.random.rand(gray.shape[0],1)

U_p, D_p, V_p = SVD_POWER(gray, 1000)

print(U_p.shape)
print(D_p.shape)
print(V_p.shape)

k = D_p.shape[0]

gray_50 = U_p[:,0:int(k/2)].dot(np.diag(D_p[0:int(k/2)])).dot(V_p[0:int(k/2)])
gray_25 = U_p[:,0:int(k/4)].dot(np.diag(D_p[0:int(k/4)])).dot(V_p[0:int(k/4)])
gray_10 = U_p[:,0:int(k*0.1)].dot(np.diag(D_p[0:int(k*0.1)])).dot(V_p[0:int(k*0.1)])
gray_5 = U_p[:,0:int(k*0.05)].dot(np.diag(D_p[0:int(k*0.05)])).dot(V_p[0:int(k*0.05)])
plt.subplot(231)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
#plt.show()
plt.title("original")
plt.axis('off')

plt.subplot(232)
plt.imshow(gray_50, cmap = plt.get_cmap('gray'))
plt.title("50%")
plt.axis('off')

plt.subplot(233)
plt.imshow(gray_25, cmap = plt.get_cmap('gray'))
plt.title("25%")
plt.axis('off')

plt.subplot(234)
plt.imshow(gray_10, cmap = plt.get_cmap('gray'))
plt.title("10%")
plt.axis('off')

plt.subplot(235)
plt.imshow(gray_5, cmap = plt.get_cmap('gray'))
plt.title("5%")
plt.axis('off')

plt.show()

original_norm = np.linalg.norm(gray)
print("Forbenius Norm Caputre")
print("50%: {}".format(np.linalg.norm(gray_50) / original_norm))
print("25%: {}".format(np.linalg.norm(gray_25) / original_norm))
print("10%: {}".format(np.linalg.norm(gray_10) / original_norm))
print("5%: {}".format(np.linalg.norm(gray_5) / original_norm))