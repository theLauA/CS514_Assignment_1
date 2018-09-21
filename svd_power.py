import numpy as np

def powerIterateStep(A, x, dif, maxloop):
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
	
def powerIterate(A, x):
	Vh = np.zeros((0,x.shape[0]))
	D_tmp = []
	A_new = A
	U = np.zeros((x.shape[0],0))
	for i in range(A.shape[1]):
		#print(i)
		v_i = powerIterateStep(A_new,x,0,100)
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
#Main
A = np.array([[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,10,0],[3,4,5,6,7,8,9,10,0,0],[4,5,6,7,8,9,10,0,0,0],[5,6,7,8,9,10,0,0,0,0],[6,7,8,9,10,0,0,0,0,0],[7,8,9,10,0,0,0,0,0,0],[8,9,10,0,0,0,0,0,0,0],[9,10,0,0,0,0,0,0,0,0],[10,0,0,0,0,0,0,0,0,0]])

#x = np.transpose(np.array([1,1,1,1,1,1,1,1,1,1]))
x = np.random.rand(10,1)
U_p, D_p, V_p = powerIterate(A, x)

#print(np.linalg.norm(A.dot(v1)))
#print(D_p)
print(V_p[1])
U,D, Vh = np.linalg.svd(A)
#print(D)
print(U_p.dot(np.diag(D_p)).dot(V_p) - A)