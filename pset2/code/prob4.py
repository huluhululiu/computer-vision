## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself


# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxWx3.
#
# Be careful about division by 0.
#
# Implement in Fourier Domain / Frankot-Chellappa

def kernpad(K,m,n):
    Ko = np.zeros((m,n))
    # m=size[0]
    # n=size[1]
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

def ntod(nrm, mask, lmda):
	H,W,C=np.shape(nrm)
	# z=np.zeros((H,W,C))
	gx=np.zeros((H,W))
	gy=np.zeros((H,W))
	fr=np.float32([[-1/9,-1/9,-1/9],[-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]])
	fx=np.float32([[0.5,0,-0.5]])
	fy=np.float32([[-0.5],[0],[0.5]])
	fx=kernpad(fx,H,W)
	fy=kernpad(fy,H,W)
	# print(fy.shape)
	fr=kernpad(fr,H,W)
	for i in range(H):
		for j in range(W):
			if mask[i][j]==1:
				if nrm[i,j,2] != 0:
					gx[i][j]=-nrm[i,j,0]/nrm[i,j,2]
					gy[i][j]=-nrm[i,j,1]/nrm[i,j,2]
	Fx=np.fft.fft2(fx)
	Fy=np.fft.fft2(fy)
	Fr=np.fft.fft2(fr)
	upperfirst=np.conjugate(Fx)*np.fft.fft2(gx)
	uppersecond=np.conjugate(Fy)*np.fft.fft2(gy)
	Fx=np.absolute(Fx)
	Fy=np.absolute(Fy)
	Fr=np.absolute(Fr)
	lower=Fx**2+Fy**2+lmda*Fr**2
	z=(upperfirst+uppersecond)/lower
	z[0,0]=0
	rez=np.real(np.fft.ifft2(z))

	return rez

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#### Main function

nrm = imread(fn('inputs/phstereo/true_normals.png'))

# Un-comment  next line to read your output instead
# nrm = imread(fn('outputs/prob3_nrm.png'))


mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-6)


# Plot 3D shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.float32(range(nrm.shape[1])),np.float32(range(nrm.shape[0])))
x = x - np.mean(x[:])
y = y - np.mean(y[:])

Zmsk = Z.copy()
Zmsk[mask == 0] = np.nan
Zmsk = Zmsk - np.nanmedian(Zmsk[:])

lim = 100
ax.plot_surface(x,-y,Zmsk, \
                linewidth=0,cmap=cm.inferno,shade=True,\
                vmin=-lim,vmax=lim)

ax.set_xlim3d(-450,450)
ax.set_ylim3d(-450,450)
ax.set_zlim3d(-450,450)

plt.show()
