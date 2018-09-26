### Default modules imported. Import more if you need to.
### DO NOT USE linalg.lstsq from numpy or scipy

import numpy as np
from skimage.io import imread, imsave

## Fill out these functions yourself


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns nrm:
#    nrm: HxWx3 Unit normal vector at each location.
#
# Be careful about division by zero at mask==0 for normalizing unit vectors.
def pstereo_n(imgs, L, mask):
    gray=np.average(imgs,axis=3)
    N,H,W,C = np.shape(imgs)
    nrm = np.zeros((H,W,C))
    for i in range(H):
        for j in range(W):
            # print(np.shape(np.transpose(L)))
            
            if mask[i][j] == 1:
                solving=np.linalg.solve(np.inner(np.transpose(L),np.transpose(L)),np.inner(np.transpose(L),gray[:,i,j]))
                mag = np.sqrt(np.sum(np.square(solving)))
                nrm[i,j,:] = solving/mag
    return nrm


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    nrm:  HxWx3 Unit normal vector at each location (from pstereo_n)
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns alb:
#    alb: HxWx3 RGB Color Albedo values
#
# Be careful about division by zero at mask==0.
def pstereo_alb(imgs, nrm, L, mask):
    N,H,W,C = np.shape(imgs)
    alb = np.zeros((H,W,C))
    imgR=np.zeros((N,H,W))
    imgG=np.zeros((N,H,W))
    imgB=np.zeros((N,H,W))
    for k in range(N):
        imgR[k,:,:]=imgs[k][:,:,0]
        imgG[k,:,:]=imgs[k][:,:,1]
        imgB[k,:,:]=imgs[k][:,:,2]
    # print(imgset[1,2,3,:])

    for i in range(H):
        for j in range(W):
            if mask[i][j] == 1:
                # print(np.dot(L,np.transpose(nrm[i,j,:])))
                # print(np.inner(L,np.transpose(nrm[i,j,:])))
                J=np.dot(L,np.transpose(nrm[i,j,:]))
                alb[i,j,0]=np.sum(np.dot(imgR[:,i,j],J)) / np.sum(J)
                alb[i,j,1]=np.sum(np.dot(imgG[:,i,j],J)) / np.sum(J)
                alb[i,j,2]=np.sum(np.dot(imgB[:,i,j],J)) / np.sum(J)
            # print(np.shape(np.transpose(L)))
                # for m in range(N):
                #     J=np.inner(L,np.transpose(nrm[i,j,:]))[m]
                #     a=np.dot(imgR[m,i,j],J)/np.dot(J,J)
                #     b=np.dot(imgG[m,i,j],J)/np.dot(J,J)
                #     c=np.dot(imgB[m,i,j],J)/np.dot(J,J)
                #     sumofR+=a
                #     sumofG+=b
                #     sumofB+=c
                # alb[i,j,0]=sumofR
                # alb[i,j,1]=sumofG
                # alb[i,j,2]=sumofB/N
                # print(alb[i,j,:])
                
    return alb
    
########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

### Light directions matrix
L = np.float32( \
                [[  4.82962877e-01,   2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,   2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,   2.58819044e-01,   8.36516261e-01],
                 [ -5.00000060e-01,   0.00000000e+00,   8.66025388e-01],
                 [ -2.58819044e-01,   0.00000000e+00,   9.65925813e-01],
                 [ -4.37113883e-08,   0.00000000e+00,   1.00000000e+00],
                 [  2.58819073e-01,   0.00000000e+00,   9.65925813e-01],
                 [  4.99999970e-01,   0.00000000e+00,   8.66025448e-01],
                 [  4.82962877e-01,  -2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,  -2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,  -2.58819044e-01,   8.36516261e-01]])


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


############# Main Program


# Load image data
imgs = []
for i in range(L.shape[0]):
    imgs = imgs + [np.float32(imread(fn('inputs/phstereo/img%02d.png' % i)))/255.]

mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = pstereo_n(imgs,L,mask)

nimg = nrm/2.0+0.5
nimg = clip(nimg * mask[:,:,np.newaxis])
imsave(fn('outputs/prob3_nrm.png'),nimg)

alb = pstereo_alb(imgs,nrm,L,mask)

alb = alb / np.max(alb[:])
alb = clip(alb * mask[:,:,np.newaxis])

imsave(fn('outputs/prob3_alb.png'),alb)
