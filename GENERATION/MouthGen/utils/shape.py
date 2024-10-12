import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import cv2
import skimage as ski

def tps_warp_preset(label_id, preset_image, type='mouth', shape_id='5'):

    #preset should have 4 channels, last channel is the alpha mask
    assert preset_image.shape[2]==4

    # label to binary mask
    label_binary = (label_id == 3) | (label_id == 23) | (label_id == 17)
    label_binary = label_binary.astype('uint8')
    if label_binary.sum()==0:
        print('No mouth')
        return -1

    # find end points
    XY = np.argwhere(label_binary>0)
    Xs = XY[:,1]
    Ys = XY[:,0]
    pleft_idx = np.argmin(Xs)
    pright_idx = np.argmax(Xs)
    xl, yl = Xs[pleft_idx], Ys[pleft_idx]
    xr, yr = Xs[pright_idx], Ys[pright_idx]
    
    # Find the mouth boundary
    contours, _ = cv2.findContours(label_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    
    # open the mouth boundary curve from end points
    contour_img = np.zeros_like(label_binary)
    for points in contour:
        x = points[0][1]
        y = points[0][0]
        contour_img[x,y]=1
    delete_radius = 5
    for i in range(xl-delete_radius,xl+delete_radius):
        for j in range(yl-delete_radius,yl+delete_radius):
            contour_img[j,i] = 0
    for i in range(xr-delete_radius,xr+delete_radius):
        for j in range(yr-delete_radius,yr+delete_radius):
            contour_img[j,i] = 0
    
    # estimate disconnected contours
    contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)!=2:
        print('Complex mouth shape')
        return -1
    
    # find bottom/top contour
    bottom_idx = -1
    max_y = -1
    for idx, cntr in enumerate(contours):
        cntr = cntr.reshape(-1,2)
        if np.max(cntr[:,1]) > max_y:
            max_y = np.max(cntr[:,1])
            bottom_idx = idx
    if bottom_idx == -1:
        print('No bottom contour')
        return -1
    top_idx = abs(1-bottom_idx)
    if len(contours)>1:
        # salient points estimation
        upper_line = contours[top_idx].reshape(-1,2)
        upper_median = np.median(upper_line, axis=0).astype('int32')
        tree_upper = cKDTree(upper_line)
        nndist, nnidx = tree_upper.query(upper_median)
        upper_contour_mid = upper_line[nnidx]
        lower_line = contours[bottom_idx].reshape(-1,2)
        lower_median = np.median(lower_line, axis=0).astype('int32')
        tree_lower = cKDTree(lower_line)
        nndist, nnidx = tree_lower.query(lower_median)
        lower_contour_mid = lower_line[nnidx]
    
        # TPS-based warping
        # ratio = (np.abs(upper_contour_mid - lower_contour_mid)[1]) / np.abs(xl-xr)
        # offset = 512*ratio
        offset = min(np.square(upper_contour_mid - lower_contour_mid)[1],100)
        src_pts = np.array([[xl,yl], upper_contour_mid, lower_contour_mid, [xr,yr]]).astype(np.float32)
        dst_pts = np.array([[0,512],[512,510],[512,512+offset],[1023,512]]).astype(np.float32)
        tps = ski.transform.ThinPlateSplineTransform()
        tps.estimate(dst_pts, src_pts)
        warped = ski.transform.warp(label_binary, tps, order=0)
        warped[warped>0] = 255

        preset = preset_image.copy()
        preset = cv2.resize(preset,(1024,1024),cv2.INTER_NEAREST)
        tps_inv = ski.transform.ThinPlateSplineTransform()
        tps_inv.estimate(src_pts, dst_pts)
        unwarped = ski.transform.warp(preset, tps_inv, order=1)
        mouth_mask = unwarped[:,:,3]==1
        mouth_mask = mouth_mask.astype('uint8')

        # fill holes in mouth mask
        preset_warped = np.zeros((1024,1024,3)).astype('uint8')
        contours, _ = cv2.findContours(mouth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        preset_warped = cv2.drawContours(preset_warped, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        shape_mask = preset_warped[:,:,0] == 255

        return shape_mask, warped, unwarped
