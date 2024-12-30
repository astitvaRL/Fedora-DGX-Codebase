from shapesimilarity import shape_similarity
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import cv2
import math

from utils.SemanticSegmentation import SemanticSegmentationAll


def resize_with_padding(im, desired_size = 1024, intepolation_mode=cv2.INTER_NEAREST):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]),intepolation_mode)
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im



join = os.path.join

save_dir = './similarity_scores/'
os.makedirs(save_dir, exist_ok=True)

# set references paths
data_root = '/mnt/users_scratch/astitva/DATA/'
image_dir_name = './shape_references' 
label_id_dir_name = './shape_references' 




label_names = sorted(os.listdir(label_id_dir_name))

labels_definition_file_path = 'label_definition.json'
semantic = SemanticSegmentationAll(labels_definition_path=labels_definition_file_path, num_classes=27)

padding = 0

NUM_POINTS = 1000

ref_images = []
ref_masks = []
ref_top_points = []
ref_bottom_points = []
ref_ratio = []
for name in tqdm(label_names):
    
    imgpath = join(image_dir_name, name)
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_LINEAR)

    labelpath = join(label_id_dir_name, name)
    gt2D = cv2.imread(labelpath)
    gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
    gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)

    label_id = semantic.colors_to_labels(gt2D)

    face_region = (label_id == 3) | (label_id == 23) | (label_id == 17)
    Xs, Ys = np.where(face_region)
    if len(Xs)==0 or len(Ys)==0:
        continue
    x_min, x_max = np.min(Xs) - padding, np.max(Xs) + padding
    y_min, y_max = np.min(Ys) - padding, np.max(Ys) + padding
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > 1024:
        x_max = 1024
    if y_max > 1024:
        y_max = 1024
    
    ratio = (y_max-y_min)/1.0
    ref_ratio.append(ratio)

    img_cropped = img[x_min:x_max, y_min:y_max,:]
    img_cropped = cv2.resize(img_cropped, (1024, 1024))

    #don't resize mask to 1024x1024
    label_id = label_id[x_min:x_max, y_min:y_max]
    # label_id = cv2.resize(label_id, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    label_id = resize_with_padding(label_id, 1024)

    mouth_mask = (label_id == 3) | (label_id == 23) | (label_id == 17)
    blur_kernel = (53,53)
    mouth_mask = cv2.GaussianBlur(mouth_mask.astype('uint8')*255, blur_kernel, 0)
    mouth_mask = mouth_mask>0
    mouth_mask = mouth_mask.astype('uint8')*255

    try:

        Xs, Ys = np.where(mouth_mask>0)
        if len(Xs)<NUM_POINTS or len(Ys)<NUM_POINTS:
            continue

        top_pixels = []
        for x,y in zip(Xs,Ys):
            if mouth_mask[x-1,y]==0:
                top_pixels.append([x,y])
        
        bottom_pixels = []
        for x,y in zip(Xs,Ys):
            if mouth_mask[x+1,y]==0:
                bottom_pixels.append([x,y])

        
        top_pixels = np.array(top_pixels)
        bottom_pixels = np.array(bottom_pixels)

    # plt.scatter(top_pixels[:,0], top_pixels[:,1])
    # plt.scatter(bottom_pixels[:,0], bottom_pixels[:,1])
    # plt.savefig('zplot.png')
    # plt.close()

        top_ridx = np.random.choice(np.arange(top_pixels.shape[0]),NUM_POINTS)
        top_points = top_pixels[top_ridx]
        bottom_ridx = np.random.choice(np.arange(bottom_pixels.shape[0]),NUM_POINTS)
        bottom_points = bottom_pixels[bottom_ridx]

        # points = (points-points.min(0)) / (points.max(0)-points.min(0))

        ref_masks.append(mouth_mask)
        ref_top_points.append(top_points)
        ref_bottom_points.append(bottom_points)
        ref_images.append(img_cropped)
    
    except:
        continue



# set all paths
data_root = '/mnt/users_scratch/astitva/DATA/'
image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
label_id_dir_name = 'AD_SegMaps/labels_7k_1024' 


label_names = sorted(os.listdir(join(data_root, label_id_dir_name)))[:10]

labels_definition_file_path = 'label_definition.json'
semantic = SemanticSegmentationAll(labels_definition_path=labels_definition_file_path, num_classes=27)

padding = 0

all_images = []
all_masks = []
all_top_points = []
all_bottom_points = []
all_ratio = []
for name in tqdm(label_names):
    imgname = name.split('_')[0] + '.png'
    imgpath = join(data_root, image_dir_name, imgname)
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_LINEAR)

    labelpath = join(data_root, label_id_dir_name, name)
    gt2D = cv2.imread(labelpath)
    gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
    gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)

    label_id = semantic.colors_to_labels(gt2D)

    face_region = (label_id == 3) | (label_id == 23) | (label_id == 17)
    Xs, Ys = np.where(face_region)
    if len(Xs)==0 or len(Ys)==0:
        continue
    x_min, x_max = np.min(Xs) - padding, np.max(Xs) + padding
    y_min, y_max = np.min(Ys) - padding, np.max(Ys) + padding
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > 1024:
        x_max = 1024
    if y_max > 1024:
        y_max = 1024

    ratio = (y_max-y_min)/1.0
    all_ratio.append(ratio)

    img_cropped = img[x_min:x_max, y_min:y_max,:]
    img_cropped = cv2.resize(img_cropped, (1024, 1024))

    #don't resize mask to 1024x1024
    label_id = label_id[x_min:x_max, y_min:y_max]
    # label_id = cv2.resize(label_id, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    label_id = resize_with_padding(label_id, 1024)

    mouth_mask = (label_id == 3) | (label_id == 23) | (label_id == 17)
    blur_kernel = (53,53)
    mouth_mask = cv2.GaussianBlur(mouth_mask.astype('uint8')*255, blur_kernel, 0)
    mouth_mask = mouth_mask>0
    mouth_mask = mouth_mask.astype('uint8')*255

    try:

        Xs, Ys = np.where(mouth_mask>0)
        if len(Xs)<NUM_POINTS or len(Ys)<NUM_POINTS:
            continue

        top_pixels = []
        for x,y in zip(Xs,Ys):
            if mouth_mask[x-1,y]==0:
                top_pixels.append([x,y])
        
        bottom_pixels = []
        for x,y in zip(Xs,Ys):
            if mouth_mask[x+1,y]==0:
                bottom_pixels.append([x,y])

        
        top_pixels = np.array(top_pixels)
        bottom_pixels = np.array(bottom_pixels)

    # plt.scatter(top_pixels[:,0], top_pixels[:,1])
    # plt.scatter(bottom_pixels[:,0], bottom_pixels[:,1])
    # plt.savefig('zplot.png')
    # plt.close()

        top_ridx = np.random.choice(np.arange(top_pixels.shape[0]),NUM_POINTS)
        top_points = top_pixels[top_ridx]
        bottom_ridx = np.random.choice(np.arange(bottom_pixels.shape[0]),NUM_POINTS)
        bottom_points = bottom_pixels[bottom_ridx]

        # points = (points-points.min(0)) / (points.max(0)-points.min(0))

        all_masks.append(mouth_mask)
        all_top_points.append(top_points)
        all_bottom_points.append(bottom_points)
        all_images.append(img_cropped)
    
    except:
        continue


# all_masks = np.array(all_masks)
# all_top_points = np.array(all_top_points)
# all_bottom_points = np.array(all_bottom_points)

for idx1 in range(len(all_top_points)):
    scores = []
    for idx2 in tqdm(range(len(ref_top_points))):
        pts1 = np.concatenate([all_top_points[idx1],all_bottom_points[idx1]])
        pts2 = np.concatenate([ref_top_points[idx2],ref_bottom_points[idx2]])
        sim_top = shape_similarity(all_top_points[idx1], ref_top_points[idx2], checkRotation=False)
        sim_bottom = shape_similarity(all_bottom_points[idx1], ref_bottom_points[idx2], checkRotation=False)
        # sim = shape_similarity(pts1, pts2, rotations=0)
        # sim = shape_similarity(pts1, pts2, checkRotation=False)
        sim = sim_top + sim_bottom
        scores.append(sim)
    
    scores = np.array(scores)
    scores_idx_sort = scores.argsort()
    print(scores[scores_idx_sort[-1]])
    fig, ax = plt.subplots(1,4, figsize=(40,10))
    max_idx_1 = scores_idx_sort[-1]
    max_idx_2 = scores_idx_sort[-2]
    max_idx_3 = scores_idx_sort[-3]
    ax[0].imshow(all_masks[idx1])
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(ref_masks[max_idx_1],cv2.COLOR_RGB2BGR))
    ax[1].set_title(scores[max_idx_1],fontsize=30)
    ax[1].axis('off')
    ax[2].imshow(cv2.cvtColor(ref_masks[max_idx_2],cv2.COLOR_RGB2BGR))
    ax[2].set_title(scores[max_idx_2],fontsize=30)
    ax[2].axis('off')
    ax[3].imshow(cv2.cvtColor(ref_masks[max_idx_3],cv2.COLOR_RGB2BGR))
    ax[3].set_title(scores[max_idx_3],fontsize=30)
    ax[3].axis('off')
    plt.savefig(join(save_dir,f'{idx1}.png'))
    plt.close()
