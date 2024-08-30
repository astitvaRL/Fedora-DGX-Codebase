from skimage import measure  # for converstion from binary masks to vector points
import shapely  # for shape analysis
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import numpy.typing as npt
import rasterio.features
from typing import List
import os
from pathlib import Path


class SAM():
    def __init__(self):
        checkpoint_pth = os.path.join(Path(__file__).parent, 'sam_vit_b_01ec64.pth')
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_pth)  # For Macbook

        self.predictor = SamPredictor(sam)
        self.mask_input = None

    def set_image(self, image_cv2: npt.NDArray[np.int32]):
        print('SAM: setting image')
        self.predictor.set_image(image_cv2)
        print('SAM: done setting image')

    def predict_from_coords(self, coords: npt.NDArray[np.int32], coord_labels: npt.NDArray[np.int32]) -> npt.NDArray[np.bool_]:
        if self.mask_input is None:
            masks, scores, logits = self.predictor.predict(
                point_coords=coords,
                point_labels=coord_labels,
            )
        else:
            masks, scores, logits = self.predictor.predict(
                point_coords=coords,
                point_labels=coord_labels,
                mask_input=self.mask_input[None, :, :]
            )
        self.mask_input = logits[np.argmax(scores), :, :]

        # get the mask with highest score
        mask = masks[np.argmax(scores), :, :]

        # get contours of polygons within the mask
        contours: List[npt.NDArray[np.float64]] = measure.find_contours(255 * mask.astype(np.uint8), 128)
        # get polygons
        polygons = []
        for contour in contours:
            polygons.append(shapely.Polygon(contour[:, [1, 0]]))  # swap x and y columns

        # get one with largest area
        polygons.sort(key=lambda x: x.area, reverse=True)
        polygon = polygons[0]

        # new mask
        mask = rasterio.features.rasterize([polygon], out_shape=mask.shape).astype(np.bool_)

        return mask

    def reset(self):
        self.mask_input = None
