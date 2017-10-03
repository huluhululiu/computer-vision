## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input color image
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian

def neighbor(X,i,j,K,sgm_i,sgm_s):
    b=np.zeros((2*K+1,2*K+1,))
    sumofB=0
    for m in range(-K,K+1,1):
        for n in range(-K,K+1,1):
            if ((i+m) < X.shape[0]) and ((i+m) >= 0) and ((j+n) < X.shape[1]) and ((j+n) >= 0):
                firstterm=np.exp(-(m*m+n*n)/(2.0*sgm_s*sgm_s))
                secondterm=np.exp(ssd(X[m+i][n+j],X[i][j],sgm_i))
                b[m+K][n+K]=firstterm*secondterm
                sumofB+=b[m+K][n+K]          
    return b,sumofB

def bfilt(X,K,sgm_s,sgm_i):
    # Placeholder
    H = np.zeros(X.shape,dtype=np.float32)
    for i in range(H.shape[0]):
        print("progress:{}/{}".format(i,H.shape[0]))
        for j in range(H.shape[1]):
            finalB=np.zeros((2*K+1,2*K+1,1))
            b,sumofB=neighbor(X,i,j,K,sgm_i,sgm_s)
            for m in range(-K,K+1,1):
                for n in range(-K,K+1,1):
                    if ((i+m) < X.shape[0]) and ((i+m) >= 0) and ((j+n) < X.shape[1]) and ((j+n) >= 0):
                        finalB[m+K][n+K]=b[m+K][n+K]/sumofB
                        H[i][j]+=finalB[m+K][n+K]*X[i+m][j+n]
    return H
def ssd(a,b,sgm_i):
    s= (a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2
    return -s/(2.0*sgm_i*sgm_i)

########################## Support code below

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img1 = np.float32(imread(fn('inputs/p4_nz1.png')))/255.
img2 = np.float32(imread(fn('inputs/p4_nz2.png')))/255.

K=9

print("Creating outputs/prob4_1_a.png")
im1A = bfilt(img1,K,2,0.5)
imsave(fn('outputs/prob4_1_a.png'),clip(im1A))


print("Creating outputs/prob4_1_b.png")
im1B = bfilt(img1,K,4,0.25)
imsave(fn('outputs/prob4_1_b.png'),clip(im1B))

print("Creating outputs/prob4_1_c.png")
im1C = bfilt(img1,K,16,0.125)
imsave(fn('outputs/prob4_1_c.png'),clip(im1C))

# Repeated application
print("Creating outputs/prob4_1_rep.png")
im1D = bfilt(img1,K,2,0.125)
for i in range(8):
    im1D = bfilt(im1D,K,2,0.125)
imsave(fn('outputs/prob4_1_rep.png'),clip(im1D))

# Try this on image with more noise    
print("Creating outputs/prob4_2_rep.png")
im2D = bfilt(img2,2,8,0.125)
for i in range(16):
    im2D = bfilt(im2D,K,2,0.125)
imsave(fn('outputs/prob4_2_rep.png'),clip(im2D))