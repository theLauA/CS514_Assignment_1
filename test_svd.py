from svd_power import *
import numpy as np

#Find SVD for Matrix A
A = np.array([[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,10,0],[3,4,5,6,7,8,9,10,0,0],[4,5,6,7,8,9,10,0,0,0],[5,6,7,8,9,10,0,0,0,0],[6,7,8,9,10,0,0,0,0,0],[7,8,9,10,0,0,0,0,0,0],[8,9,10,0,0,0,0,0,0,0],[9,10,0,0,0,0,0,0,0,0],[10,0,0,0,0,0,0,0,0,0]])

#x = np.transpose(np.array([1,1,1,1,1,1,1,1,1,1]))
x = np.random.rand(10,1)
U_p, D_p, V_p = SVD_POWER(A, x,0,1000)

#print(np.linalg.norm(A.dot(v1)))
#print(D_p)
#print(V_p[1])
U,D, Vh = np.linalg.svd(A)
#Reconstruct A and find different
print(U_p.dot(np.diag(D_p)).dot(V_p) - A)