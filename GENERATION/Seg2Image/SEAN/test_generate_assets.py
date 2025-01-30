"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import cv2
import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import time

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.status = 'UI_mode'

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# # create a webpage that summarizes the all results
# web_dir = os.path.join(opt.results_dir, opt.name,
#                        '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir,
#                     'Experiment = %s, Phase = %s, Epoch = %s' %
#                     (opt.name, opt.phase, opt.which_epoch))

# custom test
first=True
obj_dic_global = dict()
style_code_parent_dir = os.environ['STYLE_CODE_DIR']
# style_codes_dir = f'styles_test/style_codes/original.png/'
style_codes_dir = f'{style_code_parent_dir}/style_codes/original.png/'
stye_codes_mean_dir = f'styles_train/mean_style_code/mean/'
style_code_names = sorted(os.listdir(style_codes_dir))

# prepare style codes
num_classes = 11
for code_idx in range(num_classes):
    obj_dic_global[str(code_idx)] = {}
    if str(code_idx) in style_code_names:
        obj_dic_global[str(code_idx)]['ACE'] = torch.from_numpy(np.load(os.path.join(style_codes_dir, str(code_idx), 'ACE.npy'))).cuda()
    else:
        obj_dic_global[str(code_idx)]['ACE'] = torch.from_numpy(np.load(os.path.join(stye_codes_mean_dir, str(code_idx), 'ACE.npy'))).cuda()

outdir = os.environ['SEAN_OUTDIR']
for i, data_i in tqdm(enumerate(dataloader)):

    start = time.time()
    save_name = data_i['path'][0].split('/')[-1]

    if i * opt.batchSize >= opt.how_many:
        break
    
    data_i['obj_dic'] = obj_dic_global

    generated = model(data_i, mode='UI_mode')

    generated_np = generated.squeeze(0).permute(1,2,0).cpu().numpy()
    generated_np = (generated_np + 1.0) / 2.0 * 255.0
    generated_np = generated_np.astype('uint8')
    print(f"GAN synthesis took {time.time()-start} seconds")
    cv2.imwrite(f'{outdir}/{save_name}',cv2.cvtColor(generated_np, cv2.COLOR_RGB2BGR))


