import os
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

join = os.path.join

# set paths
data_root = '/mnt/users_scratch/astitva/DATA/'
labels_definition_file_path = './label_definition.json'
image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
label_id_dir_name = 'AD_SegMaps/labels_7k_1024' 
preset_dir = './presets'
preset_type = 'mouth'
preset_prompts = ['with an open mouth', 'with mouth wide open',  'frowning', 'with tongue out', 'with vampire teeth']
shape_ids = ['0','1','2','3','5']
output_root = 'output'
visualize_dir = 'visualize_plots'
os.makedirs(visualize_dir, exist_ok=True)

folders = os.listdir(output_root)

for folder in tqdm(folders):
    subdir = join(output_root, folder)
    files = sorted(os.listdir(subdir))
    fig, ax = plt.subplots(1,6, figsize=(60,10))
    fig.tight_layout()
    ax[0].imshow(Image.open(join(subdir, files[-1])))
    # ax[0].set_title('Original Image', fontsize=35)
    ax[0].axis('off')
    for idx in range(len(files)-1):
        ax[idx+1].imshow(Image.open(join(subdir, files[idx])))
        # ax[idx+1].set_title(preset_prompts[idx], fontsize=35)
        ax[idx+1].axis('off')
    save_name = (subdir.split('/')[-1]).split('_')[0] + '.png'
    plt.savefig(join(visualize_dir, save_name))
    plt.close()
