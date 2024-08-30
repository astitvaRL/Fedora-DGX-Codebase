import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import io
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision
import monai
import json
from monai.networks import one_hot

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from utils.dataset import Dataset_multiclass_face
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentation
join = os.path.join


if __name__ == '__main__':
    torch.manual_seed(999)
    np.random.seed(999)

    # set paths
    data_root = 'D:\\DATA\\Amateur Drawing Semantic Segmentations\\20240716-1034 (1)\\'
    labels_definition_file_path = join(data_root, 'label_definition.json')
    ckpt_dir = './checkpoints'
    sam_original_ckpt_path = join(ckpt_dir,'sam_original/sam_vit_b_01ec64.pth')
    image_dir_name = 'drawings_face'
    label_id_dir_name = 'labels_ID_face' 
    embed_dir_name = f"{image_dir_name}_embeddings" # precomputed image embeddings will be saved here if not saved already
    embedding_dir_path = join(data_root, embed_dir_name)
    task_name = 'animseg_face_300' # finetuned checkpoint will be saved here
    model_save_path = join(ckpt_dir, task_name)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(join(model_save_path, 'pred_seg_vis'), exist_ok=True)
    
    # training choice
    precompute_embeddings = False # False if already precomputed and saved
    resume_training = True
    resume_ckpt = join(model_save_path, 'model_best.pth')

    # prepare SAM model
    model_type = 'vit_b'

     # load original SAM cackpoint for finetuning, or load an existing checkpoint for further training
    init_checkpoint = join(sam_original_ckpt_path)
    if resume_training:
        init_checkpoint = resume_ckpt

    device = 'cuda:0'
    num_classes = 10 
    sam_model = sam_model_registry[model_type](num_classes = num_classes, checkpoint=init_checkpoint).to(device)

    # precompute image embeddings using original SAM model
    if precompute_embeddings:
        os.makedirs(embedding_dir_path, exist_ok=True)
        print('Precomputing image embeddings...')
        names = sorted(os.listdir(join(data_root, image_dir_name)))
        for name in tqdm(names):
            image_data = io.imread(join(data_root, image_dir_name, name))
            if image_data.shape[-1]>3 and len(image_data.shape)==3:
                image_data = image_data[:,:,:3]
            if len(image_data.shape)==2:
                image_data = np.repeat(image_data[:,:,None], 3, axis=-1)
            sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(image_data)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
            assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
            # precompute and save the image embedding
            with torch.no_grad():
                embedding = sam_model.image_encoder(input_image)
                np.save(join(embedding_dir_path, name.split('.png')[0]+'.npy'), embedding.cpu().numpy()[0])
        print('Image embeddings saved at -->', embedding_dir_path)

    # create dataset
    train_dataset = Dataset_multiclass_face(data_root = data_root, img_embed_dir_name = embed_dir_name, label_id_dir_name = label_id_dir_name )

    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # training config
    num_epochs = 10
    save_frequency = 1
    show_frequency= 1
    ignore_background = False
    losses = []
    best_loss = 1e10
    visualize = SemanticSegmentation(labels_definition_file_path)

    # Set up the optimizer, losses, hyperparameters
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # start training
    sam_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        # train
        for step, (image_embedding, gt, bbox) in enumerate(tqdm(train_dataloader)):
            # not computing gradients for image encoder and prompt encoder
            with torch.no_grad():
                # resize gt to 256x256
                gt = F.resize(gt, 256, torchvision.transforms.InterpolationMode.NEAREST)
                gt = torch.nn.functional.one_hot(gt.squeeze(1),num_classes)
                B,_, H, W = gt.shape
                gt = torch.permute(gt,(0,3,1,2))
                
                # resizing by a factor of 4 (1024-->256)
                bbox = bbox//4 
                bbox = bbox.to(device)            
                
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=bbox[:, None, :],
                    masks=None,
                )
            # computing gradients for mask decoder only
            mask_predictions, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=True,
            )

            # Optional, doesn't help much if mask is already being passed as prompt
            if ignore_background: 
                mask_predictions = mask_predictions*bg_mask
                gt = gt.to(device)*bg_mask
            
            loss = seg_loss(mask_predictions, gt.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch%show_frequency==0:    
            labels_out = torch.argmax(torch.Tensor(mask_predictions[0]), dim=0)
            plt.imshow(visualize.labels_to_colors(labels_out.cpu().numpy().astype('uint8')))
            plt.savefig(join(model_save_path, f"pred_seg_vis/{epoch}.png"))
            
        epoch_loss /= step
        losses.append(epoch_loss)
        
        if epoch%save_frequency==0:
            print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
            # save the latest model checkpoint
            torch.save(sam_model.state_dict(), join(model_save_path, 'model_latest.pth'))
            # save the best model checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(sam_model.state_dict(), join(model_save_path, 'model_best.pth'))

        np.save(join(model_save_path,f"loss_log_current_epoch.npy"),np.array(losses))
