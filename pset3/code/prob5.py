## Default modules imported. Import more if you need to.

import numpy as np


#########################################
### Hamming distance computation
### You can call the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2


def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################





## Fill out these functions yourself

# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
def census(img):
    W = img.shape[1]
    H = img.shape[0]
    c = np.zeros([H, W], dtype=np.uint32)
    for y in range(W):
        for x in range(H):
            cen = 0
            count=0
            for sig_y in range(-2, 3):
                for sig_x in range(-2, 3):
                    if sig_x != 0 or sig_y != 0:
                        if x + sig_x >= H or y + sig_y >= W or x + sig_x < 0 or y + sig_y < 0:
                            bit = 1
                        else:
                            cen <<=1
                            if img[x+sig_x, y+sig_y] < img[x, y]:
                                bit =1
                            else:
                                bit=0
                            cen = cen + bit
            c[x, y]=cen
    return c
    

# Given left and right image and max disparity D_max, return a disparity map
# based on matching with  hamming distance of census codes. Use the census function
# you wrote above.
#
# d[x,y] implies that left[x,y] matched best with right[x-d[x,y],y]. Disparity values
# should be between 0 and D_max (both inclusive).
def smatch(left,right,dmax):
    H, W = left.shape
    d = np.zeros(left.shape)
    census_left = census(left)
    census_right = census(right)
    difer=np.zeros((H,W,dmax+1))
    difer[...]=float('inf')
    difer[...,0]=hamdist(census_left,census_right)
    for i in range(1,dmax+1):
        difer[:,i:,i]=hamdist(census_left[:, i:], census_right[:, :-i])
    d=np.argmin(difer,axis=2)
    return d
    
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = imread(fn('inputs/left.jpg'))
right = imread(fn('inputs/right.jpg'))

d = smatch(left,right,40)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/20.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob5.png'),dimg)
