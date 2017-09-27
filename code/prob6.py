## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

def im2wv(img,nLev):
    result=[]
    unitMatrix=np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]])
    for i in range(0,nLev):
        height,width=img.shape[0],img.shape[1]
        half_h,half_w=int(height/2),int(width/2)
        LargeL=np.zeros([half_h,half_w])
        LargeH1=np.zeros([half_h,half_w])
        LargeH2=np.zeros([half_h,half_w])
        LargeH3=np.zeros([half_h,half_w])
        for m in range(0,height-1,2):
            for n in range(0,width-1,2):
                abcd=np.array([img[m][n],img[m+1][n],img[m][n+1],img[m+1][n+1]])
                lhhh=0.5*np.inner(unitMatrix,np.transpose(abcd))
                sm=int(m/2)
                sn=int(n/2)
                LargeL[sm][sn]=lhhh[0]
                LargeH1[sm][sn]=lhhh[1]
                LargeH2[sm][sn]=lhhh[2]
                LargeH3[sm][sn]=lhhh[3]
        v=[LargeH1,LargeH2,LargeH3]
        result.append(v)
        if i==nLev-1:
            result.append(LargeL)
        img=LargeL
    # Placeholder that does nothing
    return result


def wv2im(pyr):
    nLev=len(pyr)
    unitMatrix=np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]])
    Amatrix=np.linalg.inv(unitMatrix)
    newpyr=pyr[:]
    for i in range(0,nLev-1):
        LL=newpyr[nLev-1-i]
        LH1=newpyr[nLev-1-i-1][0]
        LH2=newpyr[nLev-1-i-1][1]
        LH3=newpyr[nLev-1-i-1][2]
        newIm=np.zeros([len(LL)*2,len(LL)*2])
        for m in range(np.shape(LL)[0]):
            for n in range(np.shape(LL)[1]):
                lhhh=np.array([LL[m][n],LH1[m][n],LH2[m][n],LH3[m][n]])
                abcd=2*np.inner(Amatrix,np.transpose(lhhh))
                newIm[m*2][n*2]=abcd[0]
                newIm[m*2+1][n*2]=abcd[1]
                newIm[m*2][n*2+1]=abcd[2]
                newIm[m*2+1][n*2+1]=abcd[3]
        newpyr[nLev-1-i-1]=newIm
        newpyr.pop()
    return newpyr[-1]



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


# Visualize pyramid like in slides
def vis(pyr, lev=0):
    if len(pyr) == 1:
        return pyr[0]/(2**lev)

    sz=pyr[0][0].shape
    sz1 = [sz[0]*2,sz[1]*2]
    img = np.zeros(sz1,dtype=np.float32)

    img[0:sz[0],0:sz[1]] = vis(pyr[1:],lev+1)

    # Just scale / shift gradient images for visualization
    img[sz[0]:,0:sz[1]] = pyr[0][0]*(2**(1-lev))+0.5
    img[0:sz[0],sz[1]:] = pyr[0][1]*(2**(1-lev))+0.5
    img[sz[0]:,sz[1]:] = pyr[0][2]*(2**(1-lev))+0.5

    return img



############# Main Program


img = np.float32(imread(fn('inputs/p6_inp.png')))/255.

# Visualize pyramids
pyr = im2wv(img,1)
imsave(fn('outputs/prob6a_1.png'),clip(vis(pyr)))

pyr = im2wv(img,2)
imsave(fn('outputs/prob6a_2.png'),clip(vis(pyr)))

pyr = im2wv(img,3)
imsave(fn('outputs/prob6a_3.png'),clip(vis(pyr)))

# Inverse transform to reconstruct image
im = clip(wv2im(pyr))
imsave(fn('outputs/prob6b.png'),im)

# Zero out some levels and reconstruct
for i in range(len(pyr)-1):

    for j in range(3):
        pyr[i][j][...] = 0.

    im = clip(wv2im(pyr))
    imsave(fn('outputs/prob6b_%d.png' % i),im)
