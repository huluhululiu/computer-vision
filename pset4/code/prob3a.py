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

# Copy this from solution to problem 2.
def buildcv(left,right,dmax):
    cv = 24 * np.ones([left.shape[0],left.shape[1],dmax+1], dtype=np.float32)
    H, W = left.shape
    d = np.zeros(left.shape)
    census_left = census(left)
    census_right = census(right)
    cv[...,0]=hamdist(census_left,census_right)
    for i in range(1,dmax+1):
        cost=hamdist(census_left[:, i:], census_right[:, :-i])
        if cost.any()>0:
            cv[:,i:,i]=cost
    #cv=np.argmin(cv)
    #return d
    return cv


# Implement the forward-backward viterbi method to smooth
# only along horizontal lines. Assume smoothness cost of
# 0 if disparities equal, P1 if disparity difference <= 1, P2 otherwise.
#
# Function takes in cost volume cv, and values of P1 and P2
# Return the disparity map
def viterbilr(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]

    cvbar=np.zeros([H,W,D])
    cvbar[:,0,:]=cv[:,0,:]
    s=np.zeros([H,D,D])
    z = np.zeros([H, W, D])
    resd=np.zeros([H,W], dtype=int)
    for d in range(D):
        dprime = np.arange(D)
        singleS=np.ones(dprime.shape)*P2
        singleS[np.where(dprime==d)] = 0
        singleS[np.where(np.abs(dprime-d) == 1)] = P1
        s[...,d]=np.stack([singleS for _ in range(H)],axis=0)

    for x in range(1,W):
        cvx=cvbar[:, x-1 ,:] 
        stackcvbar=np.stack([cvx for _ in range(D)],axis=1)
        z[:,x,:]=np.argmin(np.add(s,stackcvbar),axis=2)
        cvbar[:,x,:]=cv[:,x,:]+np.min(np.add(s,stackcvbar),axis=2)
    
    resd[:,-1]=np.argmin(cvbar[:,-1,:], axis=1)
    for j in reversed(range(-1,W-1)):
         resd[:, j] = np.diag(z[:, j+1, resd[:, j+1]])


    return resd
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)
                   
cv = buildcv(left_g,right_g,50)
d = viterbilr(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3a.jpg'),dimg)
