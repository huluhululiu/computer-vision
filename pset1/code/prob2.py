## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input 8-bit grayscale image
# Return equalized image with intensities from 0-255
def histeq(X):
	hist,bins = np.histogram(X.flatten(),256,normed=True)
	cdf = hist.cumsum()
	cdf = cdf * 255 / cdf[-1]
	im2 = np.interp(X.flatten(),bins[:-1],cdf)
	return cdf.reshape(X.shape)
    

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = imread(fn('inputs/p2_inp.png'))

out = histeq(img)

out = np.maximum(0,np.minimum(255,out))
out = np.uint8(out)
imsave(fn('outputs/prob2.png'),out)
