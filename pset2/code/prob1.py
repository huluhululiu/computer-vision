## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

# Copy from Pset1/Prob6 
def im2wv(img,nLev):

    return [img]


# Copy from Pset1/Prob6 
def wv2im(pyr):

    return pyr[-1]


# Fill this out
# You'll get a numpy array/image of coefficients y
# Return corresponding coefficients x (same shape/size)
# that minimizes (x - y)^2 + lmbda * abs(x)
def denoise_coeff(y,lmbda):

    return y



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program

lmain = 0.88

img = np.float32(imread(fn('inputs/p1.png')))/255.

pyr = im2wv(img,4)
for i in range(len(pyr)-1):
    for j in range(2):
        pyr[i][j] = denoise_coeff(pyr[i][j],lmain/(2**i))
    pyr[i][2] = denoise_coeff(pyr[i][2],np.sqrt(2)*lmain/(2**i))
    
im = wv2im(pyr)        
imsave(fn('outputs/prob1.png'),clip(im))
