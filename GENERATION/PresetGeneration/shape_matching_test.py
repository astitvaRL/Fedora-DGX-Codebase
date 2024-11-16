from shapesimilarity import shape_similarity
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import cv2

from utils.SemanticSegmentation import SemanticSegmentationAll


join = os.path.join

# x = np.linspace(1, -1, num=200)

# y1 = 2*x**3 + 1
# y2 = 2*x**2 + 2

# shape1 = np.column_stack((x, y1))
# shape2 = np.column_stack((x, y2))
# similarity = shape_similarity(shape1, shape2)

# plt.plot(shape1[:,0], shape1[:,1], linewidth=2.0)
# plt.plot(shape2[:,0], shape2[:,1], linewidth=2.0)

# plt.title(f'Shape similarity is: {similarity}', fontsize=14, fontweight='bold')
# plt.savefig('plot.png')



# set paths
data_root = '/mnt/users_scratch/astitva/DATA/'
image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
label_id_dir_name = 'AD_SegMaps/labels_7k_1024' 

save_dir = join(data_root, 'AD_SegMaps/mouth_clusters/')
os.makedirs(save_dir, exist_ok=True)


label_names = sorted(os.listdir(join(data_root, label_id_dir_name)))[:500]

labels_definition_file_path = 'label_definition.json'
semantic = SemanticSegmentationAll(labels_definition_path=labels_definition_file_path, num_classes=27)

padding = 0

all_masks = []
all_top_points = []
all_bottom_points = []
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

    face_region = label_id == 6
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

    img_cropped = img[x_min:x_max, y_min:y_max]
    img_cropped = cv2.resize(img_cropped, (1024, 1024))
    label_id = label_id[x_min:x_max, y_min:y_max]
    label_id = cv2.resize(label_id, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    mouth_mask = (label_id == 3) | (label_id == 23) | (label_id == 17)
    blur_kernel = (53,53)
    mouth_mask = cv2.GaussianBlur(mouth_mask.astype('uint8')*255, blur_kernel, 0)
    mouth_mask = mouth_mask>0
    mouth_mask = mouth_mask.astype('uint8')*255

    try:

        Xs, Ys = np.where(mouth_mask>0)
        if len(Xs)<500 or len(Ys)<500:
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

        top_ridx = np.random.choice(np.arange(top_pixels.shape[0]),500)
        top_points = top_pixels[top_ridx]
        bottom_ridx = np.random.choice(np.arange(bottom_pixels.shape[0]),500)
        bottom_points = bottom_pixels[bottom_ridx]

        # points = (points-points.min(0)) / (points.max(0)-points.min(0))

        all_masks.append(mouth_mask)
        all_top_points.append(top_points)
        all_bottom_points.append(bottom_points)
    
    except:
        continue


all_masks = np.array(all_masks)
all_top_points = np.array(all_top_points)
all_bottom_points = np.array(all_bottom_points)

idx1 = 10

while True:
    idx2 = np.random.randint(len(all_top_points))
    sim_top = shape_similarity(all_top_points[idx1], all_top_points[idx2], checkRotation=False)
    sim_bottom = shape_similarity(all_bottom_points[idx1], all_bottom_points[idx2], checkRotation=False)
    sim = sim_bottom
    print(sim)
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].imshow(all_masks[idx1])
    ax[0].axis('off')
    ax[1].imshow(all_masks[idx2])
    ax[1].axis('off')
    plt.savefig(f'z_sim_{sim}.png')
    plt.close()

    # breakpoint()