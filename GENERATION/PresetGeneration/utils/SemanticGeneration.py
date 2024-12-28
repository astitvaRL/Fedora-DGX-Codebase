import sys
sys.path.append('/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/GENERATION/Seg2Image/SEAN')
sys.path.append('/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/GENERATION/Seg2Image/SEAN/data')
sys.path.append('/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/GENERATION/Seg2Image/SEAN/models')

import argparse

from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
import models
import data
import torch
import torchvision.transforms as transforms
from PIL import Image


def init_parser(parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--status', type=str, default='test')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
        parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--results_dir', type=str, default='./TMP/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=1024, help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=512, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=182, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        # for setting inputs
        parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
        parser.add_argument('--dataset_mode', type=str, default='custom')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=28, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')
        # for displays
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')
        # for generator
        parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--z_dim', type=int, default=256,
                            help="dimension of the latent z vector")
        # for instance-wise features
        parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')
        return parser

def init_model(ckpt_dir, exp_name):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = init_parser(parser)
    parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
    parser.set_defaults(serial_batches=True)
    parser.set_defaults(no_flip=True)
    parser.set_defaults(phase='test')
    opt, unknown = parser.parse_known_args()
    # modify model-related parser options
    model_name = opt.model
    model_option_setter = models.get_option_setter(model_name)
    parser = model_option_setter(parser, False)
    # modify dataset-related parser options
    dataset_mode = opt.dataset_mode
    dataset_option_setter = data.get_option_setter(dataset_mode)
    parser = dataset_option_setter(parser, False)
    opt, unknown = parser.parse_known_args(['--label_dir','TMP','--image_dir','TMP','--gpu_ids','0','--no_instance','--load_size','1024','--crop_size','1024', '--label_nc','11', '--dataset_mode', 'custom'])
    opt.isTrain = False
    opt.semantic_nc = 11
    opt.checkpoints_dir = ckpt_dir
    opt.name = exp_name
    opt.batch_size = 1
    model = Pix2PixModel(opt)
    return model


def get_model(ckpt_dir='/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/GENERATION/Seg2Image/SEAN/checkpoints/', exp_name='seg2drawings'):
    model = init_model(ckpt_dir, exp_name)
    return model

def generate(model, image, label):
    # image --> float32 tensor of shape [1,3,1024,1024]
    image = (image-image.min())/(image.max()-image.min())
    image = torch.Tensor(image).float()
    image = image.permute(2,1,0).unsqueeze(0)
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image = normalize(image)
    # label --> float32 tensor of shape [1,1,1024,1024]
    label = torch.Tensor(label).float()
    label = label[None, None, :, :]
    # instance --> zero tensor
    instance = torch.Tensor([0]).long() 
    input_data = dict()
    input_data['image'] = image
    input_data['label'] = label
    input_data['instance'] = instance
    input_data['path'] = 'tmp_style_code'
    # generate
    generated = model(input_data, mode='inference')
    # tensor to numpy array
    generated_np = generated.squeeze(0).permute(1,2,0).cpu().numpy()
    generated_np = (generated_np + 1.0) / 2.0 * 255.0
    generated_np = generated_np.astype('uint8')
    return generated_np