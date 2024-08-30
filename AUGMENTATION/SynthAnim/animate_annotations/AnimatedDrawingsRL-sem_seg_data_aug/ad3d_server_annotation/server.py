import json
import signal
import sys
import time
import shutil
from pathlib import Path
import cv2
import numpy.typing as npt

import animated_drawings.streaming.pytelepathy as pytelepathy

from annotation_builder import get_character_joint_keypoints, get_character_segmentation, export_ad3d_annotations
from view import View, MockTool


running = True  # Add this global flag to control the loop

animated_drawing = None


def create_annotation_level_data(image_fn: str, fb_mask_fn: str = None, output_dir_name: str = None) -> str:

    """ prep image for annotation """
    # TODO: Nicky, modify this to reconstruct numpy array representation of image based on what Unity sends
    image_cv2 = cv2.imread(image_fn)
    if image_cv2.shape[-1] == 4:  # if image is transparent, paste onto white background and drop alpha
        alpha_zero_mask = image_cv2[:, :, 3] == 0
        image_cv2[alpha_zero_mask, :3] = [255, 255, 255]
        image_cv2 = image_cv2[:, :, :3]
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    # get keypoints
    try:
        character_joint_keypoints = get_character_joint_keypoints(image_cv2)
    except Exception as e:
        print( f'Error getting joint keypoints: {str(e)}')
        return f'Error getting joint keypoints: {str(e)}'

    # if fb mask is not specified, compute it
    if fb_mask_fn:
        fb_mask = cv2.imread(str(fb_mask_fn))
    else:
        fb_mask = get_character_segmentation(image_cv2)

    # create one part encompassing whole character
    part = {
        'Type': 'External',
        'Name': 'FullCharacter',
        'AreaMask': 'FullCharacter',
        'ParentName': None,
        'ForwardOrientation': 'None',
        'BottomSplit': None,
        'TopSplit': None
    }

    # export the annotations
    if output_dir_name is None:
        annotations_outdir = Path(image_fn).parent / Path(image_fn).stem / 'annotations'
    else:
        annotations_outdir = Path(image_fn).parent / output_dir_name / 'annotations'

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
    cv2.imwrite(f'{txtr_name}.png', cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR))

    """ Create the character rig """

    # load keypoints
    with open(annotations_outdir / "keypoints.json", 'r') as f:
        kpts = json.load(f)

    # create a 'mocktool' using keypoints, set image and parts
    try:
        tool = MockTool(kpts)
        tool.image_cv2 = image_cv2
        tool.image_name = image_fn
        tool.load_parts_from_export(annotations_outdir / "parts.json")
    except Exception as e:
        print( f'Error creating mock tool: {str(e)}')
        return f'Error creating mock tool: {str(e)}'

    # construct the 'right' view
    try:
        vr = View(tool, 'DRight')
        vr.generate_drawing_left_right_view()

        # construct the 'left' view
        vl = View(tool, 'DLeft')
        vl.generate_drawing_left_right_view()
    except Exception as e:
        print( f'Error creating views: {str(e)}')
        return f'Error creating views: {str(e)}'

    # set the rig outdir and export the views
    if output_dir_name is None:
        rig_outdir = Path(image_fn).parent / Path(image_fn).stem / 'rig'
    else:
        annotations_outdir = Path(image_fn).parent / output_dir_name / 'annotations'
        rig_outdir = Path(image_fn).parent / output_dir_name / 'rig'

    try:
        export_ad3d_annotations(vl, vr, rig_outdir)
    except Exception as e:
        print( f'Error exporting rig/annotations: {str(e)}')

    # return the location where the views have been exported
    return str(rig_outdir)


def handle_sigint(sig, frame):
    print("Interrupt signal received. Exiting the program.")
    global running
    running = False


def annotation_res_channel_callback(request_msg):
    print(request_msg)
    print(
        f"/annotation_data (received_request: {len(request_msg)} bytes): {request_msg[:16]}..."
    )
    return create_annotation_level_data(request_msg)


if __name__ == "__main__":
    # Print program name and instructions
    print("[Telepathy] AD3D Annotation Server")
    print("Press Ctrl+C to exit the program at any time.")

    # Register signal and signal handler
    signal.signal(signal.SIGINT, handle_sigint)

    host = "::"  # Default host (any IPv4 and IPv6)
    port = 8884  # Default port

    # If command line arguments are provided, use them.
    if len(sys.argv) == 3:
        host, port = sys.argv[1], int(sys.argv[2])
    else:
        print(f"Using default values:\nHost: {host}\nPort: {port}")

    config = pytelepathy.WebSocketServerConfig()
    config.host = host
    config.port = port
    config.num_max_threads = 4
    config.reconnect_interval_s = 5
    config.max_read_limit_bytes = 1024 * 1024 * 1024
    config.max_write_limit_bytes = 1024 * 1024 * 1024

    # ResChannel configuration
    annotation_res_channel = pytelepathy.ResChannelConfig()
    annotation_res_channel.path = "/annotation_data"
    annotation_res_channel.callback = annotation_res_channel_callback
    config.res_channels.append(annotation_res_channel)

    server = pytelepathy.WebSocketServer(config)
    server.start()

    # Keep the program running until SIGINT is received
    try:
        while running:
            time.sleep(1)  # Sleep to reduce CPU usage
    except KeyboardInterrupt:
        pass

    print("Client terminated gracefully.")
