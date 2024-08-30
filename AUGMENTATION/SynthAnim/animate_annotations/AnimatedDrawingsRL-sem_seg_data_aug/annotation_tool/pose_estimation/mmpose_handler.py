# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
from pathlib import Path
from mmpose.apis import (inference_top_down_pose_model, init_pose_model)


class Handler:
    def __init__(self):
        config = os.path.join(Path(__file__).parent, 'config.py')
        checkpoint = os.path.join(Path(__file__).parent, 'best_AP_epoch_72.pth')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = init_pose_model(config, checkpoint, device)

    def _preprocess(self, image_np):
        return image_np

    def _inference(self, image_cv2):
        preds, _ = inference_top_down_pose_model(self.model, image_cv2, person_results=None)
        return preds

    def _post_process(self, preds):
        output = [{'keypoints': pred['keypoints'].tolist()} for pred in preds]
        return output

    def predict_keypoints(self, image_np):
        preprocessed = self._preprocess(image_np)
        inference = self._inference(preprocessed)
        post_processed = self._post_process(inference)
        return post_processed
