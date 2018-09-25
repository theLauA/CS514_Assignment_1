Run 3_27.py with Python 2
For python notebook, run jupyter notebook with python 2 and open 3_27.ipynb

Method SVD_POWER(A, maxloop) in svd_power.py is an implementation singular value decomposition with Power Method.
It takes two paramters, A is a N by M matrix, maxloop is the maximum iteration for Power Method; returns U,D,Vh where A = U.dot(np.diag(D)).dot(Vh).
The implementation runs powerIterate_v3 on A to find v1, then runs powerIterate_v3 on A' = A - d11 * u1.dot(v1), where d11 is the first singular value, u1 is the first left singular vector, and v1 is the first right singular vector.
Repeats until min(N, M), or the resulting singular value is larger than the last singular value.

3_27.py
Part 1
Import an image with matplotlib library. Change the color image to grayscale, resulting in a 100 x 100 matrix.
Perform svd with SVD_POWER.
Reconstruct with first 50%, 25%, 10%, 5% of the singular values and singular vectors
Display the orignal and reconstructed images

Part 2
Caculate the norm of the reconstructed images and compares against original's