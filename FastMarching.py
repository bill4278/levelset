import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import heapq as hq
import copy

def compute_grad(x,methodId=0):
    if methodId==0:
        return np.sqrt(np.sum(np.square((np.array(np.gradient(x)))), axis=0))
    if methodId==1:
        grad_x = cv2.Sobel(img, -1, 1, 0, ksize=5)
        grad_y = cv2.Sobel(img, -1, 0, 1, ksize=5)
        return np.sqrt(grad_x*grad_x + grad_y*grad_y)
    if methodId==2:
        grad_img = np.abs(cv2.Laplacian(img, -1, ksize=5))
        return grad_img


INF = sys.maxsize

seeds = []

def on_EVENT_BUTTONDOWN(event, x, y, flags, param):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        seeds.append([x,y])
    if event == cv2.EVENT_RBUTTONDOWN:
        print(seeds)
        fmm_process(seeds, max_visits, seg_thresh)

def eikonal_update(u,img,dist,img_grad):
    dleft = (dist[u['id']-1] if(u['seedX']>0) else INF)
    dright = (dist[u['id']+1] if(u['seedX']<img.shape[0]-1) else INF)
    dup = (dist[u['id']-img.shape[0]] if(u['id']>=img.shape[0]) else INF)
    ddown = (dist[u['id']+img.shape[0]] if(u['id']<img.size-img.shape[0]) else INF)

    dhoriz = np.min((dleft, dright))
    dvert = np.min((dup, ddown))

    cell_val = img_grad.flatten()[u['id']]
    det = 2*dvert*dhoriz - dvert*dvert - dhoriz*dhoriz + 2*cell_val*cell_val
    if (det >=0):
        return 0.5*(dhoriz+dvert+np.sqrt(det))
    else:
        return np.min((dhoriz,dvert))+cell_val

def cell_update(u,Pque,dist,cell_status):
    if(cell_status[u['id']]==255.0): return
    estimate = eikonal_update(u,img,dist,img_grad)
    if(estimate<dist[u['id']]):
        dist[u['id']]=estimate
        if (cell_status[u['id']]==0):
            u['dist']=estimate
            Pque.append(copy.deepcopy(u))
            if(cell_status[u['id']]==0):
                 cell_status[u['id']]=1.0

def fmm_process(seeds, max_visits, seg_thresh):
    Pque = list()
    for i in range(seeds.__len__()):
        Pque.append({'id': seeds[i][0]+seeds[i][1]*img.shape[1],'seedX': seeds[i][0], 'dist':0.0})
        dist[seeds[i][0]+seeds[i][1]*img.shape[1]]=0.0;

    while ((Pque.__len__())and (max_visits)):
        max_visits-=1
        u = hq.nsmallest(1,Pque, key = lambda s:s['dist'])[0]
        Pque.remove(u)
        if(cell_status[u['id']]==255.0 or dist[u['id']]>seg_thresh):
            continue
        cell_status[u['id']] = 255.0
        u['id']-=1
        u['seedX']-=1
        if (u['seedX']>=0) : cell_update(u,Pque,dist,cell_status)

        u['id']+=2
        u['seedX']+=2
        if (u['seedX']<img.shape[0]) : cell_update(u,Pque,dist,cell_status)
        u['id']-=1
        u['seedX']-=1
        
        u['id']-=img.shape[0]
        if (u['id']>=0) : cell_update(u,Pque,dist,cell_status)

        u['id']+=2*img.shape[0]
        if (u['id']<img.size) : cell_update(u,Pque,dist,cell_status)

        if(1):
            showCellStatus = np.array(cell_status).reshape((img.shape[0], img.shape[1]))
            cv2.imshow('4', np.uint8(showCellStatus))
            cv2.waitKey(delay=1)
            # print(Pque.__len__(),max_visits)


if __name__ == "__main__":

    #img = cv2.resize(cv2.imread("./IVOCT_images/01-006-0M-LAD-1-PRE00000001.png", flags=0),(512,512))
    #img = cv2.imread("./IVUS_images/IMG-0001-00001.bmp", flags=0)
    #img = cv2.resize(cv2.imread("./test_images/97002085.11.tiff", flags=0),(384,291))
    img = cv2.imread("./test_images/lsm_ori.png", flags=0)
    cv2.imshow('1',img)

    img = np.array(img)

    #smoothed_img = cv2.GaussianBlur((img-np.mean(img)),(35,35),5)
    #v2.imshow('2',smoothed_img)

    #img_grad = compute_grad(img,1)
    img_grad = img

    cv2.imshow('2',np.uint8(255.0* img_grad / (np.max(img_grad)-np.min(img_grad))))


    output = INF*np.ones_like(img)
    dist = list(INF*np.ones(img.size))
    max_visits = -1
    seg_thresh = 20
    cell_status = list(np.zeros(img.size))
    cv2.setMouseCallback("2", on_EVENT_BUTTONDOWN)
    cv2.waitKey(0)

    
