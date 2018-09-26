## Default modules imported. Import more if you need to.

import numpy as np
from scipy.signal import convolve2d as conv2

def get_cluster_centers(im,num_clusters):
    # Implement a method that returns an initial grid of cluster centers. You should first
    # create a grid of evenly spaced centers (hint: np.meshgrid), and then use the method
    # discussed in class to make sure no centers are initialized on a sharp boundary.
    # You can use the get_gradients method from the support code below.
    cluster_centers = np.zeros((num_clusters,2),dtype='int')
    rows=im.shape[0]
    cols=im.shape[1]
    S = np.sqrt(rows * cols / num_clusters).astype(int)
    cluster_root = np.sqrt(num_clusters).astype(int)
    xc=np.linspace(np.floor(S / 2).astype(int),cols-np.floor(S/2).astype(int),cluster_root).astype(int)
    yc=np.linspace(np.floor(S / 2).astype(int),rows-np.floor(S/2).astype(int),cluster_root).astype(int)

    im_gradients = get_gradients(im)
    temp = []
    for j in xc:
        for i in yc:
            start = np.array([i -1, j -1])
            sub_grads = im_gradients[i - 1:i + 2, j - 1:j + 2]
            lowest_grad = np.unravel_index(np.argmin(sub_grads), (3, 3))
            temp.append(start + np.array(lowest_grad))
    cluster_centers = np.array(temp)
    return cluster_centers


    """ YOUR CODE GOES HERE """

def slic(im,num_clusters,cluster_centers):
    # Implement the slic function such that all pixels assigned to a label
    # should be close to each other in squared distance of augmented vectors.
    # You can weight the color and spatial components of the augmented vectors
    # differently. To do this, experiment with different values of spatial_weight.
    h,w,c = im.shape
    clusters = np.zeros((h,w))
    S = np.sqrt(h * w / num_clusters).astype(int)

    clusteriprime = np.concatenate((im[cluster_centers[:, 0], cluster_centers[:, 1], ...], cluster_centers), axis=1)

    clusterw = np.tile(clusteriprime.reshape(1, 1, num_clusters, 5), (h, w, 1, 1))

    nx, ny = np.meshgrid(np.arange(h), np.arange(w))
    augvim = np.concatenate((im, ny.reshape(h, w, 1), nx.reshape(h, w, 1)), axis=2)

    imv = np.tile(augvim.reshape(h, w, 1, 5), (1, 1, num_clusters, 1))
    spatial_weight = 5
    for i in range(25):

        clusters = np.argmin(np.where(np.logical_and(np.absolute(clusteriprime[..., 3] - imv[..., 3])
                                           <= S,np.absolute(clusterw[..., 4] - imv[..., 4]) <= S),
                            np.sum((clusterw[..., :3] - imv[..., :3]) ** 2, axis=3) +
                            np.sum( ((clusterw[..., 3:] - imv[..., 3:]) * spatial_weight) ** 2, axis=3), float('inf')), axis=2)

        for j in range(num_clusters):
            idx = np.argwhere(clusters == j)
            clusteriprime[j, :] = np.mean(augvim[idx[:, 0], idx[:, 1], :], axis=0)

    """ YOUR CODE GOES HERE """

    return clusters

########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# Use get_gradients (code from pset1) to get the gradient of your image when initializing your cluster centers.
def get_gradients(im):
    if len(im.shape) > 2:
        im = np.mean(im,axis=2)
    df = np.float32([[1,0,-1]])
    sf = np.float32([[1,2,1]])
    gx = conv2(im,sf.T,'same','symm')
    gx = conv2(gx,df,'same','symm')
    gy = conv2(im,sf,'same','symm')
    gy = conv2(gy,df.T,'same','symm')
    return np.sqrt(gx*gx+gy*gy)

# normalize_im normalizes our output to be between 0 and 1
def normalize_im(im):
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im

# create an output image of our cluster centers
def create_centers_im(im,centers):
    for center in centers:
        im[center[0]-2:center[0]+2,center[1]-2:center[1]+2] = [255.,0.,255.]
    return im

im = np.float32(imread(fn('inputs/lion.jpg')))

num_clusters = [25,49,64,81,100]
for num_clusters in num_clusters:
    cluster_centers = get_cluster_centers(im,num_clusters)
    imsave(fn('outputs/prob1a_' + str(num_clusters)+'_centers.jpg'),normalize_im(create_centers_im(im.copy(),cluster_centers)))
    out_im = slic(im,num_clusters,cluster_centers)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1,np.float32(out_im.flatten())/float(num_clusters)))[:,0:3]
    dimg = dimg.reshape([out_im.shape[0],out_im.shape[1],3])
    imsave(fn('outputs/prob1b_'+str(num_clusters)+'.jpg'),normalize_im(dimg))
