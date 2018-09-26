## Default modules imported. Import more if you need to.

import numpy as np
from scipy.signal import convolve2d as conv2

def get_cluster_centers(im,num_clusters):
    # Implement a method that returns an initial grid of cluster centers. You should first
    # create a grid of evenly spaced centers (hint: np.meshgrid), and then use the method
    # discussed in class to make sure no centers are initialized on a sharp boundary.
    # You can use the get_gradients method from the support code below.
    cluster_centers = np.zeros((num_clusters,2),dtype='int')

    """ YOUR CODE GOES HERE """
    rows,cols,channels = im.shape
    S = np.sqrt(rows*cols/num_clusters).astype(int)
    xc = np.arange(cols)
    xc = xc[np.floor(S/2).astype(int)::S]
    yc = np.arange(rows)
    yc = yc[np.floor(S/2).astype(int)::S]
    im_gradients = get_gradients(im)
    temp = []
    for j in xc:
        # end = np.array([0, j+2])
        # if j == cols-1:
        #     start[1] = j-2
        #     end[1] = j+1
        for i in yc:
            # start = np.array([i-1, j-1])
            # end[0] = i+2
            # if i == rows - 1:
            #     start[0] = i-2
            #     end[0] = i+1
            # sub_grads = im_gradients[start[0]:end[0],start[1]:end[1]]
            start = np.array([i-1, j-1])
            sub_grads = im_gradients[i-1:i+2,j-1:j+2]
            sub_opt = np.unravel_index(np.argmin(sub_grads),(3,3))
            temp.append(start + np.array(sub_opt))
    cluster_centers = np.array(temp)
    return cluster_centers

def slic(im,num_clusters,cluster_centers):
    # Implement the slic function such that all pixels assigned to a label
    # should be close to each other in squared distance of augmented vectors.
    # You can weight the color and spatial components of the augmented vectors
    # differently. To do this, experiment with different values of spatial_weight.
    h,w,c = im.shape
    clusters = np.zeros((h,w))

    """ YOUR CODE GOES HERE """
    S = np.sqrt(h*w/num_clusters).astype(int)
    #create augmented vectors
    cnt_aug = np.concatenate((im[cluster_centers[:,0], cluster_centers[:,1],...],cluster_centers),axis=1)
    nx,ny = np.meshgrid(np.arange(h),np.arange(w))
    im_aug = np.concatenate((im,ny.reshape(h,w,1),nx.reshape(h,w,1)),axis=2)
    #initializing
    im_re = np.tile(im_aug.reshape(h,w,1,5),(1,1,num_clusters,1))
    min_dists = float('inf')*np.ones((h,w))
    spatial_weight = 15
    MAX_ITER = 25
    #iterating
    for i in range(MAX_ITER):
        cnt_re = np.tile(cnt_aug.reshape(1,1,num_clusters,5),(h,w,1,1))
        ka_dists = np.where(np.logical_and(np.absolute(cnt_re[...,3]-im_re[...,3]) <= S, np.absolute(cnt_re[...,4]-im_re[...,4]) <= S), 
            np.sum((cnt_re[...,:3]-im_re[...,:3])**2,axis=3) + np.sum(((cnt_re[...,3:]-im_re[...,3:])*spatial_weight)**2,axis=3),float('inf'))
        # add previous minimal distances to the tail
#         ka_dists = np.concatenate((ka_dists, min_dists.reshape(h,w,1)), axis=2)
        # current nearest labels
        id2 = np.argmin(ka_dists,axis=2)
#         id0 = np.repeat(np.arange(h), w).reshape(h,w)
#         id1 = np.tile(np.arange(w),h).reshape(h,w)
#         min_dists = ka_dists[id0,id1,id2]
        # update clusters
#         clusters = np.where(np.logical_and(clusters != id2, id2 < num_clusters), id2, clusters)
        clusters = id2
        # update centers
        for j in range(num_clusters):
            idx = np.argwhere(clusters == j)
            cnt_aug[j, :] = np.mean(im_aug[idx[:,0],idx[:,1],:],axis=0)
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
    imsave(fn('outputs/prob1b_'+'mw'+str(num_clusters)+'.jpg'),normalize_im(dimg))
