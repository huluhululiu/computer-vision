## Default modules imported. Import more if you need to.

import numpy as np


## Fill out these functions yourself

# Fits a homography between pairs of pts
#   pts: Nx4 array of (x,y,x',y') pairs of N >= 4 points
# Return homography that maps from (x,y) to (x',y')
#
# Can use np.linalg.svd
def getH(pts): 
    N=pts.shape[0] 
    x=pts[:,0] 
    y=pts[:,1] 
    xpri=pts[:,2] 
    ypri=pts[:,3] 

    ax=np.zeros([N,9]) 
    ay=np.zeros([N,9]) 

    ax[:,0]=-x[:] 
    ax[:,1]=-y[:] 
    ax[:, 2] = 1
    ax[:, 6] = np.multiply(xpri[:], x[:])
    ax[:, 7] = np.multiply(xpri[:], y[:])
    ax[:, 8] = xpri[:]

    ay[:,3]=-x[:]
    ay[:,4]=-y[:]
    ay[:,5]=-1
    ay[:, 6] = np.multiply(ypri[:], x[:])
    ay[:, 7] = np.multiply(ypri[:], y[:])
    ay[:, 8] = ypri[:]

    P = np.zeros([2*N, 9])
    P[::2]=ax
    P[1::2]=ay
    u,d,v=np.linalg.svd(P)
    v=v.transpose()
    h=np.reshape(v[:,-1],[3,3])

    return h
    

# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Return a spliced color image.
def splice(src,dest,dpts):
    height, width, C= np.shape(dest)
    H, W, src_c = np.shape(src)
    prime_cordinates = np.array([[0, 0], [0, W - 1], [H-1, 0], [H-1, W-1]])
    pts = np.hstack((dpts, prime_cordinates))
    Hresult = getH(pts)
    allx=dpts[:,0]
    ally=dpts[:,1]
    minx,maxx = int(min(allx)),int(max(allx))
    miny,maxy = int(min(ally)),int(max(ally))
    for x in range(minx, maxx+1):
        for y in range(miny, maxy+1):
            destcoor = np.array([[x], [y], [1]])
            xtemp, ytemp, ztemp = np.dot(Hresult, destcoor)
            xtemp = xtemp/ztemp
            ytemp = ytemp/ztemp
            if xtemp >= 0 and xtemp + 1 < H and ytemp >= 0 and ytemp + 1 < W:
                dest[y, x, :] = bilinear_interp(ytemp, xtemp, src)
    return dest


def bilinear_interp(x,y,src):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, src.shape[1]-1);
    x1 = np.clip(x1, 0, src.shape[1]-1);
    y0 = np.clip(y0, 0, src.shape[0]-1);
    y1 = np.clip(y1, 0, src.shape[0]-1);

    Ia = src[ y0, x0 ]
    Ib = src[ y1, x0 ]
    Ic = src[ y0, x1 ]
    Id = src[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


simg = np.float32(imread(fn('inputs/p4src.png')))/255.
dimg = np.float32(imread(fn('inputs/p4dest.png')))/255.
dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded

comb = splice(simg,dimg,dpts)

imsave(fn('outputs/prob4.png'),comb)
