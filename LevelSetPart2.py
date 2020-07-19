import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_grad(x):
    return np.array(np.gradient(x))

def compute_norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    return 1. / (1. + compute_norm(compute_grad(x))**2)

#img = cv2.imread("./IVOCT_images/01-006-0M-LAD-1-PRE00000145.png", flags=0)
#img = cv2.imread("./IVUS_images/IMG-0001-00001.bmp", flags=0)
img = cv2.imread("./test_images/lsm_ori.png", flags=0)
cv2.imshow('1',img)

img = np.array(img)

smoothed_img = cv2.GaussianBlur((img-np.mean(img)),(31,31),10)

F = stopping_fun(smoothed_img)

cv2.imshow('2',F)

def default_phi(x):
    # Initialize surface phi at the border (5px from the border) of the image
    # i.e. 1 outside the curve, and -1 inside the curve
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = -1.
    return phi

dt = 1.0

phi = default_phi(img)
cv2.imshow('3',phi)

for i in range(500):
    dphi = compute_grad(phi)
    dphi_norm = compute_norm(dphi)

    dphi_t = F * dphi_norm

    phi = phi + dt * dphi_t 
    t_phi = np.zeros_like(phi)
    t_phi[phi>0]=255
    cv2.imshow('4',t_phi)
    cv2.waitKey(delay = 1)  
    print(i)



cv2.waitKey()