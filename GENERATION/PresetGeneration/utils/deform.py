import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from scipy.spatial import cKDTree


def get_salient_points(mask):
    # find end points
    XY = np.argwhere(mask > 0)
    Xs = XY[:, 1]
    Ys = XY[:, 0]
    pleft_idx = np.argmin(Xs)
    pright_idx = np.argmax(Xs)
    xl, yl = Xs[pleft_idx], Ys[pleft_idx]
    xr, yr = Xs[pright_idx], Ys[pright_idx]

    # Find the mouth boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]

    # open the mouth boundary curve from end points
    contour_img = np.zeros_like(mask)
    for points in contour:
        x = points[0][1]
        y = points[0][0]
        contour_img[x, y] = 1
    delete_radius = 5
    # left end point
    del_x_min = max(0, xl - delete_radius)
    del_x_max = min(1024, xl + delete_radius)
    del_y_min = max(0, yl - delete_radius)
    del_y_max = min(1024, yl + delete_radius)
    for i in range(del_x_min, del_x_max):
        for j in range(del_y_min, del_y_max):
            contour_img[j, i] = 0
    # right end point
    del_x_min = max(0, xr - delete_radius)
    del_x_max = min(1024, xr + delete_radius)
    del_y_min = max(0, yr - delete_radius)
    del_y_max = min(1024, yr + delete_radius)
    for i in range(del_x_min, del_x_max):
        for j in range(del_y_min, del_y_max):
            contour_img[j, i] = 0

    # estimate disconnected contours
    contours, _ = cv2.findContours(
        contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) > 2:
        return -1

    # find bottom/top contour
    bottom_idx = -1
    max_y = -1
    for idx, cntr in enumerate(contours):
        cntr = cntr.reshape(-1, 2)
        if np.max(cntr[:, 1]) > max_y:
            max_y = np.max(cntr[:, 1])
            bottom_idx = idx

    if bottom_idx == -1:
        return -1
    top_idx = abs(1 - bottom_idx)

    if len(contours) > 1:
        # salient points estimation
        upper_line = contours[top_idx].reshape(-1, 2)
        upper_median = np.median(upper_line, axis=0).astype("int32")
        tree_upper = cKDTree(upper_line)
        nndist, nnidx = tree_upper.query(upper_median)
        upper_contour_mid = upper_line[nnidx]
        lower_line = contours[bottom_idx].reshape(-1, 2)
        lower_median = np.median(lower_line, axis=0).astype("int32")
        tree_lower = cKDTree(lower_line)
        nndist, nnidx = tree_lower.query(lower_median)
        lower_contour_mid = lower_line[nnidx]
        return [xl, yl], [xr, yr], upper_contour_mid, lower_contour_mid
    return -1


def get_deformation_params(label_type, shape_id, label_binary, face_extremes):
    face_x_min, face_x_max, face_y_min, face_y_max = face_extremes
    mouth_pose = None
    h_mask, w_mask = label_binary.shape
    contours, _ = cv2.findContours(
        label_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    box_pts = box.copy()
    anchors = get_salient_points(label_binary)
    if anchors != -1:
        anchor_pts = np.array([a for a in anchors])
        w_len = np.abs(anchor_pts[0][0] - anchor_pts[1][0])
        h_len = np.abs(anchor_pts[2][1] - anchor_pts[3][1])
        len_ratio = None
        if w_len == 0 or h_len == 0:
            len_ratio = 0
        else:
            len_ratio = h_len / w_len


        # estimate deformation offsets
        h_offset = 0
        w_offset = 0
        bias = 0.2
        if shape_id=='0':
            if len_ratio==1.0: # no adjustment needed
                pass
            elif len_ratio<1.0: # adjust mouth height
                h_offset = abs(w_len-h_len)*(1-bias)
            else: # adjust mouth width
                w_offset = abs(w_len-h_len)*(1-bias)

        elif shape_id=='1':
            if len_ratio>5: # no adjustment needed
                pass
            else:
                w_offset = -1*abs(w_len - 5*h_len)*(1-bias)*0.2

        # elif shape_id=='2':
        #     if len_ratio>1.5: # no adjustment needed
        #         pass
        #     else:
        #         w_offset = -1*abs(1.5*w_len - 3*h_len)*(1-bias)

            # w_scale = w_len / len_ratio
            # h_scale = h_len / len_ratio



        # h_offset = ((w_len - h_len)/(w_len + h_len)) * scale_h
        # w_offset = (h_len - w_len) * scale_w
        # print("shape_id: ", shape_id, "h_offset: ", h_offset, "w_offset: ", w_offset)

        src_pts = np.concatenate([box_pts, anchor_pts])
        deformed_box_pts = box_pts.copy()
        deformed_anchor_pts = anchor_pts.copy()

        # preventing deformation to leak outside the image
        lowest_shape_point = deformed_anchor_pts[-1][1]
        padding = 10
        h_offset_final = min(h_offset, face_x_max-lowest_shape_point-padding) # x and y naming convention might be reversed

        deformed_box_pts[0][0] -= w_offset
        deformed_box_pts[2][0] -= w_offset
        deformed_box_pts[1][0] += w_offset
        deformed_box_pts[3][0] += w_offset
        deformed_box_pts[2][1] += h_offset_final
        deformed_box_pts[3][1] += h_offset_final
        deformed_anchor_pts[-1][1] += h_offset_final
        deformed_anchor_pts[1][0] -= w_offset
        deformed_anchor_pts[0][0] += w_offset

        target_pts = np.concatenate([deformed_box_pts, deformed_anchor_pts])
        if src_pts.shape[0] != 8 or target_pts.shape[0] != 8:
            print("TPS estimation failed")
            return -1
        tps = ski.transform.ThinPlateSplineTransform()
        tps.estimate(target_pts, src_pts)
        return tps, src_pts, target_pts, mouth_pose
    return -1


def tps_warp_box_mouth(
    image, label_id, preset_shape, label_type="mouth", shape_id=None
):
    metadata = {}
    # get extreme points of face region
    face_region = (label_id == 2) | (label_id == 3) | (label_id == 4) | (label_id == 5) | (label_id == 6) | (label_id == 12) | (label_id == 13) | (label_id == 17) | (label_id == 18) | (label_id == 22) | (label_id == 23)
    face_Xs, face_Ys = np.where(face_region)
    face_x_min, face_x_max = np.min(face_Xs), np.max(face_Xs)
    face_y_min, face_y_max = np.min(face_Ys), np.max(face_Ys)
    face_extremes = (face_x_min, face_x_max, face_y_min, face_y_max)
    # label to binary mask
    label_binary = (label_id == 3) | (label_id == 23) | (label_id == 17)
    label_binary = label_binary.astype("uint8")
    if label_binary.sum() == 0:
        print("No mouth region")
        return -1

    alpha_image = np.zeros((image.shape[0], image.shape[1], 3)).astype("uint8")
    alpha_image[label_binary > 0] = image[label_binary > 0]
    transparency = label_binary[:, :, None]
    transparency[label_binary > 0] = 255
    alpha_image = np.concatenate([alpha_image, transparency], axis=-1)

    deformation_params = get_deformation_params(
        label_type=label_type, shape_id=shape_id, label_binary=label_binary, face_extremes=face_extremes
    )
    if deformation_params == -1:
        print("TPS estimation failed")
        return -1

    tps, src_pts, target_pts, mouth_pose = deformation_params
    try:
        deformed = ski.transform.warp(alpha_image.astype("float32"), tps)
    except ValueError:
        print("TPS deformation failed")
        return -1

    return deformed, label_binary, mouth_pose


# ===================================================================================================
# ===================================================================================================
# ===================================================================================================


# ===================================================================================================
# ===================================================================================================
# ===================================================================================================


def tps_warp_preset_mouth(
    image, label_id, preset_shape, label_type="mouth", return_metadata=False
):

    metadata = {}

    # preset should have 4 channels, last channel is the alpha mask
    assert preset_shape.shape[2] == 4

    # label to binary mask
    label_binary = (label_id == 3) | (label_id == 23) | (label_id == 17)
    label_binary = label_binary.astype("uint8")
    if label_binary.sum() == 0:
        print("No mouth region")
        return -1

    salient_points = get_salient_points(label_binary)
    if salient_points != -1:
        left_end, right_end, upper_contour_mid, lower_contour_mid = salient_points
        # TPS-based warping
        # ratio = (np.abs(upper_contour_mid - lower_contour_mid)[1]) / np.abs(xl-xr)
        # offset = 512*ratio
        offset = min(np.square(upper_contour_mid - lower_contour_mid)[1], 100)
        src_pts = np.array(
            [left_end, upper_contour_mid, lower_contour_mid, right_end]
        ).astype(np.float32)
        dst_pts = np.array(
            # [[0, 512], [512, 510], [512, 512 + offset], [1023, 512]]
            [[0, 512], [512, 510], [512, 600], [1023, 512]]
        ).astype(np.float32)
        tps = ski.transform.ThinPlateSplineTransform()
        tps.estimate(dst_pts, src_pts)
        warped = ski.transform.warp(label_binary, tps, order=0)
        warped[warped > 0] = 255

        preset = preset_shape.copy()
        preset = cv2.resize(preset, (1024, 1024), cv2.INTER_NEAREST)
        tps_inv = ski.transform.ThinPlateSplineTransform()
        tps_inv.estimate(src_pts, dst_pts)
        unwarped = ski.transform.warp(preset.astype("float32"), tps_inv, order=1)
        roi_mask = unwarped[:, :, 3] == 255
        roi_mask = roi_mask.astype("uint8")

        # fill holes in roi mask
        preset_warped = np.zeros((1024, 1024, 3)).astype("uint8")
        preset_contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        preset_warped = cv2.drawContours(
            preset_warped,
            preset_contours,
            -1,
            color=(255, 255, 255),
            thickness=cv2.FILLED,
        )
        shape_mask = preset_warped[:, :, 0] == 255

        preset_warped_salient_points = get_salient_points(shape_mask.astype("uint8"))
        if preset_warped_salient_points != -1:
            left_end, right_end, upper_contour_mid, lower_contour_mid = (
                preset_warped_salient_points
            )
            target_pts = np.array(
                [left_end, upper_contour_mid, lower_contour_mid, right_end]
            ).astype(np.float32)
            tps = ski.transform.ThinPlateSplineTransform()
            tps.estimate(src_pts, target_pts)
            alpha_image = np.zeros((image.shape[0], image.shape[1], 3)).astype("uint8")
            alpha_image[label_binary > 0] = image[label_binary > 0]
            transparency = label_binary[:, :, None]
            transparency[label_binary > 0] = 255
            alpha_image = np.concatenate([alpha_image, transparency], axis=-1)
            deformed = ski.transform.warp(alpha_image.astype("float32"), tps, order=1)

            if return_metadata:
                # metadata["left_end"] = left_end
                # metadata["right_end"] = right_end
                # metadata["upper_contour_mid"] = upper_contour_mid
                # metadata["lower_contour_mid"] = lower_contour_mid
                metadata["tps"] = tps
                metadata["tps_inv"] = tps_inv
                metadata["preset_warped"] = preset_warped

            return deformed, shape_mask, label_binary, metadata

    return -1


def tps_warp_single_eye(label_binary, preset_shape):

    # find end points
    XY = np.argwhere(label_binary > 0)
    Xs = XY[:, 1]
    Ys = XY[:, 0]
    pleft_idx = np.argmin(Xs)
    pright_idx = np.argmax(Xs)
    xl, yl = Xs[pleft_idx], Ys[pleft_idx]
    xr, yr = Xs[pright_idx], Ys[pright_idx]

    # Find the roi boundary
    contours, _ = cv2.findContours(
        label_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour = contours[0]

    # open the roi boundary curve from end points
    contour_img = np.zeros_like(label_binary)
    for points in contour:
        x = points[0][1]
        y = points[0][0]
        contour_img[x, y] = 1
    delete_radius = 5
    for i in range(xl - delete_radius, xl + delete_radius):
        for j in range(yl - delete_radius, yl + delete_radius):
            contour_img[j, i] = 0
    for i in range(xr - delete_radius, xr + delete_radius):
        for j in range(yr - delete_radius, yr + delete_radius):
            contour_img[j, i] = 0

    # estimate disconnected contours
    contours, _ = cv2.findContours(
        contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) != 2:
        print("Complex mouth shape")
        return -1

    # find bottom/top contour
    bottom_idx = -1
    max_y = -1
    for idx, cntr in enumerate(contours):
        cntr = cntr.reshape(-1, 2)
        if np.max(cntr[:, 1]) > max_y:
            max_y = np.max(cntr[:, 1])
            bottom_idx = idx
    if bottom_idx == -1:
        print("No bottom contour")
        return -1
    top_idx = abs(1 - bottom_idx)
    if len(contours) > 1:
        # salient points estimation
        upper_line = contours[top_idx].reshape(-1, 2)
        upper_median = np.median(upper_line, axis=0).astype("int32")
        tree_upper = cKDTree(upper_line)
        nndist, nnidx = tree_upper.query(upper_median)
        upper_contour_mid = upper_line[nnidx]
        lower_line = contours[bottom_idx].reshape(-1, 2)
        lower_median = np.median(lower_line, axis=0).astype("int32")
        tree_lower = cKDTree(lower_line)
        nndist, nnidx = tree_lower.query(lower_median)
        lower_contour_mid = lower_line[nnidx]

        # TPS-based warping
        # ratio = (np.abs(upper_contour_mid - lower_contour_mid)[1]) / np.abs(xl-xr)
        # offset = 512*ratio
        # offset = min(np.square(upper_contour_mid - lower_contour_mid)[1],100)
        src_pts = np.array(
            [[xl, yl], upper_contour_mid, lower_contour_mid, [xr, yr]]
        ).astype(np.float32)
        dst_pts = np.array([[0, 512], [512, 180], [512, 900], [1023, 512]]).astype(
            np.float32
        )
        tps = ski.transform.ThinPlateSplineTransform()
        tps.estimate(dst_pts, src_pts)
        warped = ski.transform.warp(label_binary, tps, order=0)
        warped[warped > 0] = 255

        preset = preset_shape.copy()
        preset = cv2.resize(preset, (1024, 1024), cv2.INTER_NEAREST)
        tps_inv = ski.transform.ThinPlateSplineTransform()
        tps_inv.estimate(src_pts, dst_pts)
        unwarped = ski.transform.warp(preset.astype("float32"), tps_inv, order=1)
        roi_mask = unwarped[:, :, 3] == 255
        # roi_mask = roi_mask.astype('uint8')

        # # fill holes in roi mask
        # preset_warped = np.zeros((1024,1024,3)).astype('uint8')
        # contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # preset_warped = cv2.drawContours(preset_warped, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        # shape_mask = preset_warped[:,:,0] == 255

        return roi_mask, label_binary, warped, unwarped


def tps_warp_preset_eyes(label_id, preset_shape, label_type="eyes"):

    # preset should have 4 channels, last channel is the alpha mask
    assert preset_shape.shape[2] == 4

    # label to binary mask
    label_binary = (label_id == 4) | (label_id == 22)
    label_binary = label_binary.astype("uint8")
    if label_binary.sum() == 0:
        print("No eyes")
        return -1

    # extract left/right eyes
    num_comps, comps_im = cv2.connectedComponents(label_binary)
    if num_comps != 3:
        print("Complex eye shape!")
        return -1

    left_mask = (comps_im == 1).astype("uint8")
    right_mask = (comps_im == 2).astype("uint8")

    left_shape_mask = tps_warp_single_eye(left_mask, preset_shape)[0]
    right_shape_mask = tps_warp_single_eye(right_mask, preset_shape)[0]

    shape_mask = left_shape_mask | right_shape_mask

    return shape_mask, label_binary, left_shape_mask, right_shape_mask
