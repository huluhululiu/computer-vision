### EDF --- An Autograd Engine for instruction
## (based on joint discussions with David McAllester)

import numpy as np
import im2col as ic
from scipy.signal import convolve2d as conv2d

# Global list of different kinds of components
ops = []
params = []
values = []
velocities = []

# Global forward
def Forward():
    for c in ops: c.forward()

# Global backward    
def Backward(loss):
    for c in ops:
        c.grad = np.zeros_like(c.top)
    for c in params:
        c.grad = np.zeros_like(c.top)

    loss.grad = np.ones_like(loss.top)
    for c in ops[::-1]: c.backward() 

# SGD
def SGD(lr):
    for p in params:
        p.top = p.top - lr*p.grad


## Fill this out        
def init_momentum():
    global velocities
    velocities = [np.zeros_like(p.top) for p in params]

## Fill this out
def momentum(lr,mom=0.9):
    global velocities
    # for i in range(len(params)):
    #     velocities[i][...,1:] = params[i].grad[...,1:] + mom*velocities[i][...,0:-1]
    #     velocities[i][...,0] = params[i].grad[...,0] + mom*velocities[i][...,-1]
    #     params[i].top -= lr*velocities[i]
    for i in range(len(params)):
        velocities[i] = params[i].grad+ mom*velocities[i]
        params[i].top -= lr*velocities[i]


###################### Different kinds of nodes

# Values (Inputs)
class Value:
    def __init__(self):
        values.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()

# Parameters (Weights we want to learn)
class Param:
    def __init__(self):
        params.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()


### Operations

# Add layer (x + y) where y is same shape as x or is 1-D
class add:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = self.x.top + self.y.top

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad

        if self.y in ops or self.y in params:
            if len(self.y.top.shape) < len(self.grad.shape):
                ygrad = np.sum(self.grad,axis=tuple(range(len(self.grad.shape)-1)))
            else:
                ygrad= self.grad
            self.y.grad = self.y.grad + ygrad

# Matrix multiply (fully-connected layer)
class matmul:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = np.matmul(self.x.top,self.y.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.matmul(self.y.top,self.grad.T).T
        if self.y in ops or self.y in params:
            self.y.grad = self.y.grad + np.matmul(self.x.top.T,self.grad)


# Rectified Linear Unit Activation            
class RELU:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.maximum(self.x.top,0)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad * (self.top > 0)


# Reduce to mean
class mean:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.mean(self.x.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.top) / np.float32(np.prod(self.x.top.shape))



# Soft-max + Loss (per-row / training example)
class smaxloss:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        y = self.x.top
        y = y - np.amax(y,axis=1,keepdims=True)
        yE = np.exp(y)
        yS = np.sum(yE,axis=1,keepdims=True)
        y = y - np.log(yS); yE = yE / yS

        truey = np.int64(self.y.top)
        self.top = -y[range(len(truey)),truey]
        self.save = yE

    def backward(self):
        if self.x in ops or self.x in params:
            truey = np.int64(self.y.top)
            self.save[range(len(truey)),truey] = self.save[range(len(truey)),truey] - 1.
            self.x.grad = self.x.grad + np.expand_dims(self.grad,-1)*self.save
        # No backprop to labels!    

# Compute accuracy (for display, not differentiable)        
class accuracy:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        truey = np.int64(self.y.top)
        self.top = np.float32(np.argmax(self.x.top,axis=1)==truey)

    def backward(self):
        pass


# Downsample by 2    
class down2:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = self.x.top[:,::2,::2,:]

    def backward(self):
        if self.x in ops or self.x in params:
            grd = np.zeros_like(self.x.top)
            grd[:,::2,::2,:] = self.grad
            self.x.grad = self.x.grad + grd


# Flatten (conv to fc)
class flatten:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = np.reshape(self.x.top,[self.x.top.shape[0],-1])

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.top.shape)
            
# Convolution Layer
## Fill this out

class conv2:

    def __init__(self,x,k,s):
        ops.append(self)
        self.x = x
        self.k = k
        self.stride = s

    def forward(self):
        s = self.stride
        # calculate input/output dimensions
        batches,h_image,w_image,cin = self.x.top.shape
        h_filter,w_filter,cin,cout = self.k.top.shape
        h_out = int((h_image - h_filter)/s + 1)
        w_out = int((w_image - w_filter)/s + 1)

        # convert input image batches to columns
        self.X_col = ic.im2col_indices(self.x.top, h_filter, w_filter, s)
        K_col = self.k.top.transpose(3,2,0,1).reshape(cout,-1)
        # proceed convolution via dot product
        cols = K_col @ self.X_col
        self.top = cols.reshape(cout, h_out, w_out, batches).transpose(3,1,2,0)

    def backward(self):
        s = self.stride
        h_filter,w_filter,cin,cout = self.k.top.shape
        cols = self.grad.transpose(3,1,2,0).reshape(cout, -1)
        # print(self.grad.shape)
        # input('conv2 back')
        if self.x in ops or self.x in params:
            # pass
            k_reshape = self.k.top.transpose(3,2,0,1).reshape(cout,-1)
            temp_x = cols.T @ k_reshape
            # print(temp_x.shape)
            self.x.grad += ic.col2im_indices(temp_x.T, self.x.top.shape, h_filter, w_filter,s)
        
        if self.k in ops or self.k in params:
            temp_k = cols @ self.X_col.T
            self.k.grad += temp_k.reshape(cout, cin, h_filter, w_filter).transpose(2,3,1,0)