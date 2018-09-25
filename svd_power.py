import numpy as np
#from sklearn.preprocessing import normalize
from scipy.linalg import orth

# Both dif and maxloop are stopping criteria
# Power Method for first EigenVector
# Assume A.T dot A is provided as Matrix
def powerIterate(A, x, dif, maxloop):
	curr_dif = 1.0
	count = 0
	x_i = x
	
	while count < maxloop:
		x_tmp = np.dot(A, x_i)
		x_tmp = x_tmp / np.linalg.norm(x_tmp)
		#curr_dif = np.linalg.norm(x_i - x_tmp, 2)
		curr_dif = np.linalg.norm(A.dot(x_tmp))
		x_i = x_tmp
		count+=1
		print(curr_dif, count)
		
	return x_i

#Power Method with othornormal basis
def powerIterate_v2(A, x, maxloop):
	curr_dif = 1.00
	count = 0
	x_i = orth(x)
	while count < maxloop:
		x_tmp = np.dot(A,x_i)
		x_tmp = orth(x_tmp)
		#curr_dif = np.linalg.norm(x_i - x_tmp, 2)
		x_i = x_tmp
		count+=1
	
	return x_i

#Find v1 for matrix A 
def powerIterate_v3(A, maxloop):
	v_i = np.ones((A.shape[1],1))
	B = A.T.dot(A)
	count = 0
	while count < maxloop:
		v_i_new = B.dot(v_i)
		v_i_new = v_i_new / np.linalg.norm(v_i_new)
		
		if(np.linalg.norm(v_i-v_i_new)==0):
			break
			
		v_i = v_i_new
		count+=1
	return v_i
	
def SVD_POWER(A, maxloop):
	N, M = A.shape
	Vh = np.zeros((0,M))
	D = np.zeros(0)
	A_new = A
	U = np.zeros((N,0))
	k = min(N, M)
	for i in range(k):
		#print(i)
		v_i = powerIterate_v3(A_new,maxloop)
		A_v = A.dot(v_i)
		d_ii = np.linalg.norm(A_v)
		u_i = A.dot(v_i) / d_ii
		if i > 0:
			if d_ii > D[-1]:
				break;
		
		#print(v_i[np.newaxis].T.shape)
		Vh = np.append(Vh, v_i.T, axis=0)
		D = np.append(D,d_ii)
		U = np.append(U, u_i,axis =1)
		
		A_new = A_new - d_ii * u_i.dot(v_i.T)
		#A_new = A_new - d_ii * np.outer(u_i,v_i)
	return U, D, Vh
