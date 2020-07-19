import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_grad(x):
    return np.array(np.gradient(x))

def compute_norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    #return 1. / (1. + compute_norm(compute_grad(x))**2)
    return 1. / (1. + np.exp(-1.0*(compute_norm(compute_grad(x))-0.5)/-0.16))


def default_phi(x):
    # Initialize surface phi at the border (5px from the border) of the image
    # i.e. 1 outside the curve, and -1 inside the curve
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = -1.
    return phi

#img = cv2.resize(cv2.imread("./IVOCT_images/01-006-0M-LAD-1-PRE00000001.png", flags=0),(256,256))
#img = cv2.imread("./IVUS_images/IMG-0001-00001.bmp", flags=0)
#img = cv2.resize(cv2.imread("./test_images/97002085.11.tiff", flags=0),(384,291))
img = cv2.imread("./test_images/lsm_ori.png", flags=0)
cv2.imshow('1',img)

img = np.array(img)
smoothed_img = cv2.GaussianBlur((img-np.mean(img)),(35,35),5)

g = stopping_fun(smoothed_img)

cv2.imshow('2',g)


phi = default_phi(img)

cv2.imshow('3',phi)

def curvature(f):
    fy, fx = compute_grad(f)
    norm = np.sqrt(fx**2 + fy**2)
    Nx = fx / (norm + 1e-8)
    Ny = fy / (norm + 1e-8)
    return div(Nx, Ny)


def div(fx, fy):
    fyy, fyx = compute_grad(fy)
    fxy, fxx = compute_grad(fx)
    return fxx + fyy


def dot(x, y, axis=0):
    return np.sum(x * y, axis=axis)

v=100.0
dt=0.002

dg = compute_grad(g)

for i in range(1500):
    dphi = compute_grad(phi)
    dphi_norm = compute_norm(dphi)
    divergence = curvature(phi)

    smoothing = g * divergence * dphi_norm
    balloon = g * dphi_norm * v
    attachment = dot(dphi, dg)
    dphi_t = smoothing + balloon + attachment
    phi = phi + dt * dphi_t
    t_phi = np.zeros_like(phi)
    t_phi[phi>0]=1.0
    cv2.imshow('4',np.uint8(img*(1.0-t_phi)))
    cv2.waitKey(delay = 10)  
    print(i)



cv2.waitKey()