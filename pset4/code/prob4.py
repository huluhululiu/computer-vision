## Default modules imported. Import more if you need to.
import numpy as np
from scipy.signal import convolve2d as conv2

# Use these as the x and y derivative filters
fx = np.float32([[1,0,-1]]) * np.float32([[1,1,1]]).T / 6.
fy = fx.T


# Compute optical flow using the lucas kanade method
# Use the fx, fy, defined above as the derivative filters
# and compute derivatives on the average of the two frames.
# Also, consider (x',y') values in a WxW window.
# Return two image shape arrays u,v corresponding to the
# horizontal and vertical flow.


def lucaskanade(f1,f2,W):
    u = np.zeros(f1.shape)
    v = np.zeros(f1.shape)
    i,j=f1.shape
    midf=(f1+f2)/2
    Ix = conv2(midf, fx, mode='same')
    Iy = conv2(midf, fy, mode='same')
    It = f2-f1

    window=np.ones([W,W])

    eps=1e-10

    a=conv2(Ix**2,window,mode='same')+eps
    b=conv2(Ix*Iy,window,mode='same')
    c=conv2(Iy**2,window,mode='same')+eps

    m=conv2(Ix*It,window,mode='same')
    n=conv2(Iy*It,window,mode='same')


    v=(n*a-m*b)/(b**2-a*c)
    u=(-m-b*v)/a





    return u,v
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


f1 = np.float32(imread(fn('inputs/frame10.jpg')))/255.
f2 = np.float32(imread(fn('inputs/frame11.jpg')))/255.

u,v = lucaskanade(f1,f2,11)


# Display quiver plot by downsampling
x = np.arange(u.shape[1])
y = np.arange(u.shape[0])
x,y = np.meshgrid(x,y[::-1])
plt.quiver(x[::8,::8],y[::8,::8],u[::8,::8],-v[::8,::8],pivot='mid')

plt.show()
