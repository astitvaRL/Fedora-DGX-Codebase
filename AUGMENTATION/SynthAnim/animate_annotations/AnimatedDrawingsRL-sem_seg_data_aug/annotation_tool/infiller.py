from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import torch
from sys import platform
import numpy as np
import numpy.typing as npt
import cv2


class Infiller:
    def __init__(self, infill_blend_radius: int = 8):
        if platform == 'linux':
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                revision="fp16",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to('cuda')

        self.infill_morph_radius = infill_blend_radius

    def infill(self, prompt: str, image_np: npt.NDArray[np.uint8], mask_np: npt.NDArray[np.bool_]) -> Image.Image:

        # convert to format expected by stable diffusion
        image = Image.fromarray(image_np)
        mask_image = Image.fromarray(np.stack([255 * mask_np.astype(np.uint8),
                                               255 * mask_np.astype(np.uint8),
                                               255 * mask_np.astype(np.uint8)], axis=2))

        # padd the mask_image using self.infill_morph_radius
        kernel = np.ones((self.infill_morph_radius, self.infill_morph_radius), np.uint8)
        mask_np = np.array(mask_image)
        mask_image_with_morph_radius = cv2.dilate(mask_np, kernel, iterations=1)

        # generate the infill_image
        if platform == 'linux':
            # TODO: See how blank improves as a function of increasing kernel size
            mask_image = Image.fromarray(mask_image_with_morph_radius)
            infill_image = self.pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
            infill_image = cv2.resize(np.array(infill_image), np.array(image).shape[:2][::-1])
        else:
            print('infill called on non-linux machine. Return original image with white where infill should be')
            infill_image = np.array(image)
            infill_image[mask_image_with_morph_radius[:, :, 0] == 255] = [128, 128, 128]
            infill_image = Image.fromarray(infill_image)

        # create blending weights between original image and infill image
        blend_weights = np.zeros(mask_np.shape, dtype=np.float32)
        for idx in range(self.infill_morph_radius):
            kernel = np.ones((idx, idx), np.uint8)
            blend_weights += cv2.dilate(mask_np, kernel, iterations=1) / (self.infill_morph_radius)

        blended_image = ((255 - blend_weights) / 255) * np.array(image) + (blend_weights/255) * np.array(infill_image)
        blended_image[blend_weights == 0] = 0  # only have values around the original input mask
        alpha_channel = ((blend_weights != 0).astype(np.uint8) * 255)[:, :, 0]  # make everything transparent that isn't where mask or blending was
        part_infill = cv2.merge((blended_image.astype(np.uint8), alpha_channel))
        return Image.fromarray(part_infill)
