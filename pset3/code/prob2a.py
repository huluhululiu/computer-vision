## Default modules imported. Import more if you need to.

import numpy as np


## Fill out these functions yourself

# Fits a line iteratively
#   points: Nx2 array of (x,y) pairs of N points
#   eps: Error Threshold
#   numit: Number of iterations
#
# Return a vector L = (m,b) such that y= mx + b.
#
# Fit to minimize  the sum of \|y - mx - b\|^2 over an inlier set, where the
# inlier set is defined as points where the above square error is less than eps.
#
# In iteration 1, all points should be inliers, so that if this function is called
# with numit=1, you should return the best fit over ALL points.
#
# Should you ever hit an inlier set with fewer than 2 elements, simply return the current estimate of L.
def fitLine(points, eps, numit=10):
    original=np.copy(points)
    for i in range(numit):
        x=points[:,0]
        y=points[:,1]
        m,c=getmc(x,y)
        mask = (m*original[:, 0]+c-original[:, 1])**2 < eps
        points=original[mask,...]
        # inlier=np.concatenate(points)

    return np.float32([m,c])

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
onz1 = np.float32(rs.uniform(0,1,(1000,1)) < 0.1)
onz2 = np.float32(rs.uniform(0,1,(1000,1)) < 0.5)

# Only Gaussian Noise
pts1 = np.concatenate((x,y+0.025*gnz),axis=1)

# Different percentage of outliers
pts2 = np.concatenate((x,y+(0.025 + 50.0 * onz1)*gnz),axis=1)
pts3 = np.concatenate((x,y+(0.025 + 50.0 * onz2)*gnz),axis=1)


# Run code and plot errors
eps=0.01

ax=plt.subplot(221)
estL = fitLine(pts1,eps,1)
print("(Top Left) No outliers, simple fit Error = %.2f" % vizErr(ax,pts1, trueL, estL))

ax=plt.subplot(222)
estL = fitLine(pts2,eps,1)
print("(Top Right) 10pc outliers, simple fit Error = %.2f" % vizErr(ax,pts2, trueL, estL))

ax=plt.subplot(223)
estL = fitLine(pts2,eps,10)
print("(Bottom Left) 10pc outliers, 10 iters fit Error = %.2f" % vizErr(ax,pts2, trueL, estL))

ax=plt.subplot(224)
estL = fitLine(pts3,eps,10)
print("(Bottom Right) 50pc outliers, 10 iters fit Error = %.2f" % vizErr(ax,pts3, trueL, estL))


plt.show()
