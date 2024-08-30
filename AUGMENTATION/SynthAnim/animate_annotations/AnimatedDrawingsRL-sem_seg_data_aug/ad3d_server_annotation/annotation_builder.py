from __future__ import annotations
import json
import numpy as np
import numpy.typing as npt
import sys
from skimage import measure
from scipy import ndimage
import cv2
from keypoint import Keypoint
import shutil
from pathlib import Path
from view import View, MockTool
import yaml
import fileinput
from parts import InternalPart, ExternalPart
import logging
import requests

from pose_estimation.mmpose_handler import Handler as PoseEstimationHandler
from segmentation.sam import SAM


def get_character_joint_keypoints(image_cv2: npt.NDArray, use_docker_pose_estimation: bool = None):

    if use_docker_pose_estimation:  # pose estimation model is in docker
        data_file = {'data': cv2.imencode('.png', image_cv2)[1].tobytes()}
        resp = requests.post("http://localhost:8080/predictions/drawn_humanoid_pose_estimator", files=data_file, verify=False)
        if resp is None or resp.status_code >= 300:
            raise Exception(f"Failed to get skeletons, please check if the 'docker_torchserve' is running and healthy, resp: {resp}")
        pose_results = json.loads(resp.content)
        if len(pose_results) != 1:
            raise Exception(f"AD pose detector did not detect exactly one figure in the scene: {pose_results}")

    else:  # without docker
        pose_estimation_handler = PoseEstimationHandler()
        pose_results = pose_estimation_handler.predict_keypoints(image_cv2)
        if len(pose_results) != 1:
            raise Exception(f"AD pose detector did not detect exactly one figure in the scene: {pose_results}")

    kpts = np.array(pose_results[0]['keypoints'])

    """ create keypoints and build hierarchy """

    character_joint_keypoints = []
    character_joint_keypoints.append(Keypoint([round(x) for x in (kpts[11, :2]+kpts[12, :2])/2], c=(kpts[11, 2]+kpts[12, 2])/2, name='root', parent=None))
    character_joint_keypoints.append(Keypoint([round(x) for x in (kpts[11, :2]+kpts[12, :2])/2], c=(kpts[11, 2]+kpts[12, 2])/2, name='hip', parent='root'))
    character_joint_keypoints.append(Keypoint([round(x) for x in (kpts[5, :2]+kpts[6, :2])/2  ], c=(kpts[5, 2]+kpts[6, 2])/2, name='torso', parent='hip'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[0, :2]             ], c=kpts[0, 2], name='neck'          , parent='torso'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[6, :2]             ], c=kpts[6, 2], name='right_shoulder', parent='torso'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[8, :2]             ], c=kpts[8, 2], name='right_elbow'   , parent='right_shoulder'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[10, :2]            ], c=kpts[10, 2], name='right_hand'    , parent='right_elbow'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[5, :2]             ], c=kpts[5, 2], name='left_shoulder' , parent='torso'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[7, :2]             ], c=kpts[7, 2], name='left_elbow'    , parent='left_shoulder'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[9, :2]             ], c=kpts[9, 2], name='left_hand'     , parent='left_elbow'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[12, :2]            ], c=kpts[12, 2], name='right_hip'     , parent='root'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[14, :2]            ], c=kpts[14, 2], name='right_knee'    , parent='right_hip'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[16, :2]            ], c=kpts[16, 2], name='right_foot'    , parent='right_knee'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[11, :2]            ], c=kpts[11, 2], name='left_hip'      , parent='root'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[13, :2]            ], c=kpts[13, 2], name='left_knee'     , parent='left_hip'))
    character_joint_keypoints.append(Keypoint([round(x) for x in kpts[15, :2]            ], c=kpts[15, 2], name='left_foot'     , parent='left_knee'))

    _dict = {}
    for k in character_joint_keypoints:
        _dict[k.name] = k

    # set parents and children
    for k in character_joint_keypoints:
        parent_str = k.parent
        if parent_str is not None:
            k.parent = _dict[parent_str]
            _dict[parent_str].children.append(k)

    return character_joint_keypoints


def get_character_segmentation(img: np.ndarray, use_sam_for_segmentation: bool = None):
    if use_sam_for_segmentation:
        sam = SAM()
        sam.set_image(image_cv2)

        coords = np.array([[kpt.x, kpt.y] for kpt in character_joint_keypoints])
        labels = np.array([True for kpt in character_joint_keypoints])
        mask = 255 * sam.predict_from_coords(coords, labels)

        return mask
    else:
        return image_processing_segment(img)


def image_processing_segment(img: np.ndarray):

    """ threshold """
    img = np.min(img, axis=2)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 8)
    img = cv2.bitwise_not(img)

    """ morphops """
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=2)

    """ floodfill """
    mask = np.zeros([img.shape[0]+2, img.shape[1]+2], np.uint8)
    mask[1:-1, 1:-1] = img.copy()

    # im_floodfill is results of floodfill. Starts off all white
    im_floodfill = np.full(img.shape, 255, np.uint8)

    # choose 10 points along each image side. use as seed for floodfill.
    h, w = img.shape[:2]
    for x in range(0, w-1, 10):
        cv2.floodFill(im_floodfill, mask, (x, 0), 0)
        cv2.floodFill(im_floodfill, mask, (x, h-1), 0)
    for y in range(0, h-1, 10):
        cv2.floodFill(im_floodfill, mask, (0, y), 0)
        cv2.floodFill(im_floodfill, mask, (w-1, y), 0)

    # make sure edges aren't character. necessary for contour finding
    im_floodfill[0, :] = 0
    im_floodfill[-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, -1] = 0

    """ retain largest contour """
    mask2 = cv2.bitwise_not(im_floodfill)
    mask = None
    biggest = 0

    contours = measure.find_contours(mask2, 0.0)
    for c in contours:
        x = np.zeros(mask2.T.shape, np.uint8)
        cv2.fillPoly(x, [np.int32(c)], 1)
        size = len(np.where(x == 1)[0])
        if size > biggest:
            mask = x
            biggest = size

    if mask is None:
        msg = 'Found no contours within image'
        logging.critical(msg)
        assert False, msg

    mask = ndimage.binary_fill_holes(mask).astype(int)
    mask = 255 * mask.astype(np.uint8)

    return mask.T


def export_ad3d_annotations(vl, vr, output_dir):

    parent_outdir = output_dir
    try:
        shutil.rmtree(parent_outdir)
    except Exception:
        pass
    parent_outdir.mkdir(exist_ok=False, parents=True)

    parent_dict = {
        'root': None,
        'hip': 'root',
        'torso': 'hip',
        'neck': 'torso',
        'right_shoulder': 'torso',
        'right_elbow': 'right_shoulder',
        'right_hand': 'right_elbow',
        'left_shoulder': 'torso',
        'left_elbow': 'left_shoulder',
        'left_hand': 'left_elbow',
        'right_hip': 'root',
        'right_knee': 'right_hip',
        'right_foot': 'right_knee',
        'left_hip': 'root',
        'left_knee': 'left_hip',
        'left_foot': 'left_knee',
    }

    # create the left and right version of the character
    for view, view_str in [(vl, 'left'), (vr, 'right')]:

        # make the directory that will contain this version of the character
        outdir = parent_outdir / view_str
        outdir.mkdir(exist_ok=False, parents=True)

        # using parent dictionary, create skeleton
        skeleton = []
        for k in view.keypoints:
            skeleton.append({
                'loc': [k.x, k.y],
                'name': k.name,
                'parent': parent_dict[k.name]
            })

        # since the mesh that will be manipulated via keypoints and ARAP is only comprised of background parts, get mask and txtr that is only those pieces 
        external_parts_mask, external_parts_txtr = view.get_external_parts_mask_and_txtr()

        # the config file contents
        char_cfg = {
            'height': external_parts_mask.shape[0],
            'width': external_parts_mask.shape[1],
            'skeleton': skeleton,
            'mesh_type': 'cardboard',
            'rig_type': '2D',
            'deformer_type': 'arap_igarashi',
            'footorientation_right': None,
            'footorientation_left': None
        }

        # export char_cfg, texture, and mask
        char_cfg_fn = f'{outdir}/char_cfg.yaml'
        with open(char_cfg_fn, 'w') as f:
            yaml.dump(char_cfg, f)
        cv2.imwrite(f'{outdir}/texture_original.png', cv2.cvtColor(external_parts_txtr, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(f'{outdir}/texture_front.png', cv2.cvtColor(external_parts_txtr, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(f'{outdir}/texture_back.png', np.full([char_cfg['height'], char_cfg['width'], 3], 255, dtype=np.uint8))
        cv2.imwrite(f'{outdir}/mask.png', 255*external_parts_mask.astype(np.uint8))

        masks_dir = f'{outdir}/masks'
        Path(masks_dir).mkdir(exist_ok=False, parents=False)

        texture_dir = f'{outdir}/texture'
        Path(texture_dir).mkdir(exist_ok=False, parents=False)

        with open(f'{outdir}/parts_info.yaml', 'w') as f:
            parts = []
            for part in view.parts:
                if type(part) is InternalPart:
                    parts.append({
                        'type': 'internal',
                        'does_rdtwp': part.does_rdtwp,
                        # 'rotation_drives_flip': part.rotation_drives_flip,
                        'forward_orientation': part.flip_as_drawn,
                        'hide_on_backside': part.hide_on_backside,
                        'hide_outside_part': part.hide_outside_parent,
                        'name': part.name,
                        'parent_name': part.parent_name,
                        'view_left_transform': part.transforms['left'].tolist(),
                        'view_right_transform': part.transforms['right'].tolist(),
                        'bounding_box': part.bounding_box,
                        'point_approximation': part.point,
                    })
                if type(part) is ExternalPart:
                    parts.append({
                        'type': 'external',
                        'name': part.name,
                        'top_split': part.top_split,
                        'bottom_split': part.bottom_split,
                        'parent_name': part.parent_name,
                        'forward_orientation': part.forward_orientation,
                    })

                # assert no parts have visible textures on the very end of the image
                assert len(part.mask[ 0, :][part.mask[ 0, :] == True]) == 0, f'error: part has visible pixels on border: {part.name}'
                assert len(part.mask[-1, :][part.mask[-1, :] == True]) == 0, f'error: part has visible pixels on border: {part.name}'
                assert len(part.mask[:,  0][part.mask[:,  0] == True]) == 0, f'error: part has visible pixels on border: {part.name}'
                assert len(part.mask[:, -1][part.mask[:, -1] == True]) == 0, f'error: part has visible pixels on border: {part.name}'
                cv2.imwrite(f'{masks_dir}/{part.name}.png', 255 * part.mask.astype(np.uint8))

                # write out texture
                try:
                    cv2.imwrite(f'{texture_dir}/{part.name}.png', cv2.cvtColor(part.texture, cv2.COLOR_RGB2BGR))
                except Exception:
                    print(f'no texture_rgba for part: {part.name}')

            yaml.dump(parts, f)

    # create the mvc and run command for this character
    template_path = 'template_mvc.yaml'
    copy_path = f'{parent_outdir}/mvc.yaml'
    shutil.copy(template_path, copy_path)
    print(f'MVC Config: {copy_path}')

    with fileinput.FileInput(copy_path, inplace=True) as file:
        for line in file:
            line = line.replace('<<CHARACTER_CONFIG_PATH>>', str(Path(char_cfg_fn).parent.parent.resolve()))
            print(line, end='')

    run_cmd = f'python render.py {copy_path}'
    with open(f'{parent_outdir}/run_cmd.txt', 'w') as f:
        f.write(run_cmd)


if __name__ == '__main__':

    # parse args
    image_fn = sys.argv[1]  # get image
    use_docker_pose_estimation = sys.argv[2] == 'True'  # use docker or not
    use_sam_for_segmentation = sys.argv[3] == 'True'  # use SAM or not

    # ready image
    image_cv2 = cv2.imread(image_fn, cv2.IMREAD_UNCHANGED)
    if image_cv2.shape[-1] == 4:  # if image is transparent, paste onto white background
        alpha_zero_mask = image_cv2[:, :, 3] == 0
        image_cv2[alpha_zero_mask, :3] = [255, 255, 255]
        image_cv2 = image_cv2[:, :, :3]  # drop alpha
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    # get keypoints
    character_joint_keypoints = get_character_joint_keypoints(image_cv2, use_docker_pose_estimation)

    # # get segmentation via SAM
    # TODO: add later

    # # get segmentation via image processing
    fb_mask = get_character_segmentation(image_cv2, use_sam_for_segmentation)

    # create one part for whole character
    part = {
        'Type': 'External',
        'Name': 'FullCharacter',
        'AreaMask': 'FullCharacter',
        'ParentName': None,
        'ForwardOrientation': 'None',
        'BottomSplit': None,
        'TopSplit': None
    }

    """ export the annotations """
    annotations_outdir = Path(image_fn).parent / Path(image_fn).stem / 'annotations'
    try:
        shutil.rmtree(annotations_outdir)
    except Exception:
        pass
    annotations_outdir.mkdir(parents=True)

    # export part
    with open(annotations_outdir / "parts.json", 'w') as f:
        json.dump([part], f)

    # write out keypoints
    with open(annotations_outdir / "keypoints.json", 'w') as f:
        json.dump([[k.name, k.x, k.y] for k in character_joint_keypoints], f)

    # create mask dir and write out mask
    masks_dir = annotations_outdir / 'masks'
    masks_dir.mkdir()
    mask_name = masks_dir / part['Name']
    cv2.imwrite(f'{mask_name}.png', fb_mask)

    # create txtr dir and write out txtrs
    txtrs_dir = annotations_outdir / 'txtrs'
    txtrs_dir.mkdir()
    txtr_name = txtrs_dir / part['Name']
    cv2.imwrite(f'{txtr_name}.png', image_cv2)

    """ Create the character rig """

    with open(annotations_outdir / "keypoints.json", 'r') as f:
        kpts = json.load(f)
    tool = MockTool(kpts)

    original_img_fn = sys.argv[1]
    tool.image_cv2 = cv2.cvtColor(cv2.imread(original_img_fn), cv2.COLOR_BGR2RGB)
    tool.image_name = Path(original_img_fn).stem

    tool.load_parts_from_export(annotations_outdir / "parts.json")

    vr = View(tool, 'DRight')
    vr.generate_drawing_left_right_view()

    vl = View(tool, 'DLeft')
    vl.generate_drawing_left_right_view()

    rig_outdir = Path(image_fn).parent / Path(image_fn).stem / 'rig'
    export_ad3d_annotations(vl, vr, rig_outdir)
