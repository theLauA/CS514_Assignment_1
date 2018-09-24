import numpy as np
import math

def create_sheath(num=0):
	if num == 0:
		num=100
	data = np.zeros((3,num))
	for i in range(num):
		x = math.sin((math.pi/100)*i)
		y = math.sqrt(1-math.pow(x,2))
		z = 0.003*i
		data[:,i]=[x,y,z]
	distance = np.zeros( (3,10) )
	for i in range(5):
		distance[:,i] = data[:,i] - data[:,5]
		distance[:,i+5] = data[:,i+5] - data[:,5]
		
	return data, distance




data, distance = create_sheath()
print(data.shape)
print(distance.shape)

from svd_power import *
U, D, Vh = SVD_POWER(distance, 1000)
#U, D, Vh = np.linalg.svd(distance, full_matrices=False)
print(D)
print(Vh.T)