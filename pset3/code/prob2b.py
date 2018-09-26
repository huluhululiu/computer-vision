## Default modules imported. Import more if you need to.

import numpy as np


## Fill out these functions yourself

# Fits a line with Ransac
#   points: Nx2 array of (x,y) pairs of N points
#   K: Number of Samples per run
#   N: Number of runs
#   eps: Error Threshold for outlier
# 
# Return a vector L = (m,b) such that y= mx + b.
#
# Error is same as part a, (y-mx-b)^2
#
# Use ransac to do N runs of:
#   - Sample K points
#   - Fit line to those K points
#   - Find inlier set as those where error is < epsilon
#
# Then pick the solution whose inlier set is the largest,
# and the final solution is the best fit over this inlier set.
#
# You can use np.random.choice to sample indices of points.
def ransac(points, K, N, eps):
    leftElement = 0
    resM, resC = 0, 0
    for i in range(N):
        chooseSet = points[np.random.randint(0,points.shape[0],K)]
        x=chooseSet[:,0]
        y=chooseSet[:,1]
        m,c=getmc(x,y)
        # N=x.shape[0]
        # bigx=np.sum(x)
        # bigy=np.sum(y)
        # bigA=np.sum(np.square(x))
        # bigC=np.sum(np.square(y))
        # bigB=np.sum(np.multiply(x,y))
        # m=(N*bigB-bigx*bigy)/(N*bigA-bigx**2)
        # c=(bigy-m*bigx)/N
        mask = (m*points[:, 0]+c-points[:, 1])**2 < eps
        leftset = points[mask,...]
        if leftset.shape[0]>leftElement:
            leftElement=leftset.shape[0]
            xfin=leftset[:,0]
            yfin=leftset[:,1]
            resM,resC = getmc(xfin,yfin)




    return np.float32([resM,resC])

def getmc(x,y):
    N=x.shape[0]
    bigx=np.sum(x)
    bigy=np.sum(y)
    bigA=np.sum(np.square(x))
    bigC=np.sum(np.square(y))
    bigB=np.sum(np.multiply(x,y))
    m=(N*bigB-bigx*bigy)/(N*bigA-bigx**2)
    c=(bigy-m*bigx)/N
    return m,c




########################## Support code below

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Visualize
def vizErr(ax,pts,trueL,estL):
    x = pts[:,0]
    y = pts[:,1]

    ax.hold(True)
    ax.scatter(x,y,s=0.5,c='k')

    x0 = np.float32([np.min(x),np.max(x)])
    y0 = trueL[0]*x0+trueL[1]
    y1 = estL[0]*x0+estL[1]

    ax.plot(x0,y0,c='g')
    ax.plot(x0,y1,c='r')

    g = np.abs(y0[1]-np.sum(y0)/2)
    ax.set_ylim([np.mean(y0)-10*g,np.mean(y0)+10*g])

    return np.sum((y0-y1)**2)/2


rs = np.random.RandomState(0) # Repeatable experiments

# True line and noise free points
trueL = np.float32([1.5,3])
x = rs.uniform(-0.5,0.5,(1000,1))
y = x*trueL[0]+trueL[1]

##### Noisy measurements
# Gaussian Noise
gnz = rs.normal(0,1,(1000,1))
# Outlier Noise
onz = np.float32(rs.uniform(0,1,(1000,1)) < 0.5)

pts = np.concatenate((x,y+(0.025 + 50.0 * onz)*gnz),axis=1)


# Run code and plot errors
eps=0.01 

print("Run this code multiple times!")

ax=plt.subplot(221)
estL = ransac(pts,50,4,eps)
print("(Top Left) K = 50, N = 4, Error = %.2f" % vizErr(ax,pts, trueL, estL))

ax=plt.subplot(222)
estL = ransac(pts,5,40,eps)
print("(Top Right) K = 5, N = 40, Error = %.2f" % vizErr(ax,pts, trueL, estL))

ax=plt.subplot(223)
estL = ransac(pts,50,100,eps)
print("(Bottom Left) K = 50, N = 100, Error = %.2f" % vizErr(ax,pts, trueL, estL))

ax=plt.subplot(224)
estL = ransac(pts,5,1000,eps)
print("(Bottom Right) K = 5, N = 1000, Error = %.2f" % vizErr(ax,pts, trueL, estL))
a,b,c,d=0,0,0,0
for i in range(200):
    a+=vizErr(ax,pts, trueL,ransac(pts,50,4,eps))
    b+=vizErr(ax,pts, trueL,ransac(pts,5,40,eps))
    c+=vizErr(ax,pts, trueL,ransac(pts,50,100,eps))
    d+=vizErr(ax,pts, trueL,ransac(pts,5,1000,eps))
print(a)
print(b)
print(c)
print(d)

plt.show()
