## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2


# Fill this out
def kernpad(K,size):
    Ko = np.zeros(size,dtype=np.float32)
    m=size[0]
    n=size[1]
    a=K.shape[0]
    b=K.shape[1]
    #topleft
    Ko[0:int(a/2+1),0:int(b/2+1)]=K[int(a/2):a,int(b/2):b]
    #topright
    Ko[0:int(a/2+1),n-int(b/2):n]=K[int(a/2):a,0:int(b/2)]
    #botleft
    Ko[m-int(a/2):m,0:int(b/2+1)]=K[0:int(a/2),int(b/2):b]
    #botright
    Ko[m-int(a/2):m,n-int(b/2):n]=K[0:int(a/2),0:int(b/2)]
    return Ko

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p5_inp.png')))/255.

# Create Gaussian Kernel
x = np.float32(range(-21,22))
x,y = np.meshgrid(x,x)
G = np.exp(-(x*x+y*y)/2/9.)
G = G / np.sum(G[:])


# Traditional convolve
v1 = conv2(img,G,'same','wrap')

# Convolution in Fourier domain
G = kernpad(G,img.shape)
v2f = np.fft.fft2(G)*np.fft.fft2(img)
v2 = np.real(np.fft.ifft2(v2f))

# Stack them together and save
out = np.concatenate([img,v1,v2],axis=1)
out = np.minimum(1.,np.maximum(0.,out))

imsave(fn('outputs/prob5.png'),out)


                 
