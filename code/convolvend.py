import numpy as np
from numpy import random as rd
from scipy.signal import convolve2d as conv2
from scipy.signal import fftconvolve as fconv
from scipy.ndimage.filters import convolve as conv

batch_size = 6
img_rows = 100
img_cols = 100

k_rows = 5
k_cols = 5

in_ch = 3
out_ch = 5

images = 255*rd.rand(batch_size, img_rows, img_cols, in_ch)
filters = rd.rand(k_rows, k_cols, in_ch, out_ch)

def convSlow(imgs, flts):
	B,imgr,imgc,inc = imgs.shape
	kr,kc,inc,outc = flts.shape
	
	res = np.zeros((B,imgr-kr+1,imgc-kc+1,outc))
	for b in range(B):
		for c in range(outc):
			for i in range(inc):
				res[b,...,c] += conv2(imgs[b,...,i],flts[...,i,c],mode='valid')
	return res
	
def convOC(imgs, flts):
	B,imgr,imgc,inc = imgs.shape
	kr,kc,inc,outc = flts.shape
	res = np.zeros((B,imgr-kr+1,imgc-kc+1,outc))
	for c in range(outc):
		temp = fconv(imgs, flts[...,c].reshape(1,kr,kc,inc), mode='valid')
		res[...,c] = np.squeeze(temp)
	return res
	
def convAll(imgs, flts):
	B,imgr,imgc,inc = imgs.shape
	kr,kc,inc,outc = flts.shape
	res = np.zeros((B,imgr-kr+1,imgc-kc+1,outc))
	temp_x = imgs.reshape(B,imgr,imgc,inc,1)
	temp_x = np.repeat(temp_x,outc,axis=4)
	temp_k = flts.reshape(1,kr,kc,inc,outc)
	res = conv(temp_x,temp_k)
	return res[:,kr//2:-kr//2+1,kc//2:-kc//2+1,1,:]
	
res1 = convSlow(images,filters)
res2 = convOC(images,filters)
res3 = convAll(images,filters)
print(np.sum(np.absolute(res1-res2)))
print(np.sum(np.absolute(res1-res3)))