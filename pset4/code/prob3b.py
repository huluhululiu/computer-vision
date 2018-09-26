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




# Do SGM. First compute the augmented / smoothed cost volumes along 4
# directions (LR, RL, UD, DU), and then compute the disparity map as
# the argmin of the sum of these cost volumes. 
def SGM(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]

    cvbarup=np.zeros([H,W,D])
    cvbardown=np.zeros([H,W,D])
    cvbarleft=np.zeros([H,W,D])
    cvbarright=np.zeros([H,W,D])
    cvbarup[-1,:,:]=cv[-1,:,:]
    cvbardown[0,:,:]=cv[0,:,:]
    cvbarright[:,0,:]=cv[:,0,:]
    cvbarleft[:,-1,:]=cv[:,-1,:]
    shori=np.zeros([H,D,D])
    sverti=np.zeros([W,D,D])

    resd=np.zeros([H,W], dtype=int)

    for d in range(D):
        dprime = np.arange(D)
        singleS=np.ones(dprime.shape)*P2
        singleS[np.where(dprime==d)] = 0
        singleS[np.where(np.abs(dprime-d) == 1)] = P1
        shori[...,d]=np.stack([singleS for _ in range(H)],axis=0)

    for d in range(D):
        dprime = np.arange(D)
        singleS=np.ones(dprime.shape)*P2
        singleS[np.where(dprime==d)] = 0
        singleS[np.where(np.abs(dprime-d) == 1)] = P1
        sverti[...,d]=np.stack([singleS for _ in range(W)],axis=0)

    for x in range(1,W):
        cvx=cvbarright[:, x-1 ,:] 
        stackcvbar=np.stack([cvx for _ in range(D)],axis=1)
        cvbarright[:,x,:]=cv[:,x,:]+np.min(np.add(shori,stackcvbar),axis=2)
    for x in range(W-1,0):
        cvx=cvbarleft[:, x+1 ,:] 
        stackcvbar=np.stack([cvx for _ in range(D)],axis=1)
        cvbarleft[:,x,:]=cv[:,x,:]+np.min(np.add(shori,stackcvbar),axis=2)
    
    for x in range(1,H):
        cvx=cvbardown[x-1, : ,:] 
        stackcvbar=np.stack([cvx for _ in range(D)],axis=1)
        cvbardown[x,:,:]=cv[x,:,:]+np.min(np.add(sverti,stackcvbar),axis=2)
    for x in range(H-1,0):
        cvx=cvbardown[x+1, : ,:] 
        stackcvbar=np.stack([cvx for _ in range(D)],axis=1)
        cvbardown[x,:,:]=cv[x,:,:]+np.min(np.add(sverti,stackcvbar),axis=2)

    resd=cvbardown + cvbarup+cvbarleft+cvbarright
    




    return np.argmin(resd,axis=2)

    
    
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
d = SGM(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3b.jpg'),dimg)
