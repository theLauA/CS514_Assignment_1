import numpy as np
#from sklearn.preprocessing import normalize
from scipy.linalg import orth

# Both dif and maxloop are stopping criteria
# Power Method for first EigenVector
def powerIterate(A, x, dif, maxloop):
	curr_dif = 1.0
	count = 0
	x_i = x
	
	while curr_dif >  dif and count < maxloop:
		x_tmp = np.dot(A, x_i)
		x_tmp = x_tmp / np.linalg.norm(x_tmp)
		curr_dif = np.linalg.norm(x_i - x_tmp, 2)
		x_i = x_tmp
		count+=1
	
	return x_i

#Power Method with othornormal basis
def powerIterate_v2(A, x, dif, maxloop):
	curr_dif = 1.00
	count = 0
	x_i = orth(x)
	while dif > dif or count < maxloop:
		x_tmp = np.dot(A,x_i)
		x_tmp = orth(x_tmp)
		curr_dif = np.linalg.norm(x_i - x_tmp, 2)
		x_i = x_tmp
		count+=1
	
	return x_i


def SVD_POWER(A, x, diff, maxloop):
	Vh = np.zeros((0,x.shape[0]))
	D_tmp = []
	A_new = A
	U = np.zeros((x.shape[0],0))
	for i in range(A.shape[1]):
		#print(i)
		v_i = powerIterate(A_new,x,diff,maxloop)
		d_ii = v_i.T.dot(A).dot(v_i)
		#print(v_i)
		#d_ii = np.linalg.norm(A.dot(v_i))
		if d_ii == 0:
			pass
		
		Vh = np.append(Vh, v_i.T, axis=0)
		D_tmp = np.append(D_tmp,d_ii)
		U = np.append(U, A.dot(v_i)/d_ii,axis =1)
		
		A_new = A_new - d_ii * v_i.dot(v_i.T)
		
	return U, D_tmp, Vh
