## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

from scipy import ndimage
# Different thresholds to try
T0 = 0.5
T1 = 1.0
T2 = 1.5


########### Fill in the functions below

# Return magnitude, theta of gradients of X
def grads(X):
    #placeholder
    H = np.zeros(X.shape,dtype=np.float32)
    theta = np.zeros(X.shape,dtype=np.float32)
    dx = [[1,0,-1],[2,0,-2],[1,0,-1]]
    dy = [[1,2,1],[0,0,0],[-1,-2,-1]]
    gradX = conv2(X, dx,mode='same',boundary='symmetric')
    gradY = conv2(X, dy,mode='same',boundary='symmetric')
    theta = np.arctan2(gradY,gradX)
    H = gradX * np.cos(theta) + gradY * np.sin(theta)
    return H,theta

def nms(E,H,theta):
    #placeholder
    
    Enew=np.zeros(E.shape,dtype='bool')
    for i in range(Enew.shape[0]):
        for j in range(Enew.shape[1]):
            if ((j+1) < Enew.shape[1]) and ((j-1) >= 0) and ((i+1) < Enew.shape[0]) and ((i-1) >= 0):
                if (theta[i][j] <= -7/8 * np.pi and theta[i][j] >= 7/8*np.pi) or (theta[i][j] <= 1/8*np.pi and theta[i][j] >= -1/8*np.pi):
                    if H[i][j] <H[i][j+1] or H[i][j] <H[i][j-1]:
                        E[i][j]=0

                elif (theta[i][j] >= -7/8*np.pi and theta[i][j] <= -5/8*np.pi) or (theta[i][j] >= 1/8*np.pi and theta[i][j] <= 3/8*np.pi):
                    if H[i][j] <H[i - 1][j - 1] or H[i][j] <H[i + 1][j + 1]:
                        E[i][j] = 0
        # 90 degrees
                elif (theta[i][j] >= -5/8 *np.pi and  theta[i][j] <= -3/8*np.pi) or (theta[i][j] >= 3/8*np.pi and  theta[i][j] <= 5/8*np.pi):
                    if H[i][j] <H[i - 1][j] or H[i][j] <H[i + 1][j]:
                        E[i][j] = 0
        # 135 degrees
                else:
                    if H[i][j] <H[i - 1][j + 1] or H[i][j] <H[i + 1][j - 1]:
                        E[i][j] = 0
                    
    #Ere=np.ones(E.shape,dtype='bool')



    #hori_comp=np.where(np.logical_or((H[horizontal_index] < H[np.add(horizontal_index, h1)]), (H[horizontal_index] < H[np.add(horizontal_index, h2)])))
    #verti_comp=np.where(np.logical_or((H[vertical_index] < H[np.add(vertical_index, v1)]), (H[vertical_index] < H[np.add(vertical_index, v2)])))
    #diaplus_comp=np.where(np.logical_or((H[diag_plus_index] < H[np.add(diag_plus_index, dp1)]), (H[diag_plus_index] < H[np.add(diag_plus_index, dp2)])))
    #diagneg_comp=np.where(np.logical_or((H[diag_neg_index] < H[np.add(diag_neg_index, dg1)]), (H[diag_neg_index] < H[np.add(diag_neg_index, dg2)])))

    return E


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p3_inp.png')))/255.

H,theta = grads(img)
imsave(fn('outputs/prob3_a.png'),H/np.max(H[:]))

## Part b

E0 = np.float32(H > T0)
E1 = np.float32(H > T1)
E2 = np.float32(H > T2)

imsave(fn('outputs/prob3_b_0.png'),E0)
imsave(fn('outputs/prob3_b_1.png'),E1)
imsave(fn('outputs/prob3_b_2.png'),E2)

E0n = nms(E0,H,theta)
E1n = nms(E1,H,theta)
E2n = nms(E2,H,theta)

imsave(fn('outputs/prob3_b_nms0.png'),E0n)
imsave(fn('outputs/prob3_b_nms1.png'),E1n)
imsave(fn('outputs/prob3_b_nms2.png'),E2n)

