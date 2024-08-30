# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import sys
import asyncio
import os
import natsort
from tqdm import tqdm
import yaml


def start(user_mvc_cfg_fn: str):

    # build cfg
    from animated_drawings.config import Config
    cfg: Config = Config(user_mvc_cfg_fn)

    # create view
    from animated_drawings.view.view import View
    view = View.create_view(cfg.view)

    # create scene
    from animated_drawings.model.scene import Scene
    scene = Scene(cfg.scene, view.camera)

    scene.add_child(view.camera)

    # create controller
    from animated_drawings.controller.controller import Controller
    controller = Controller.create_controller(cfg.controller, scene, view)

    if cfg.controller.mode == 'streaming':
        print("Characters are all solved. Starting to stream...")
        asyncio.run(controller.run())
    else:
        controller.run()


if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', level=logging.DEBUG)

    SEG_DATA_DIR = "D:\\DATA\\Amateur Drawing Semantic Segmentations\\20240716-1034 (1)\\labels_resized\\"

    MVC_DIR = "labels_animated\\" # need to be relative to the current working directory

    seg_files = natsort.natsorted(os.listdir(SEG_DATA_DIR))

    FAIL_COUNT = 0
    for filename in tqdm(seg_files):
        print(filename)
        try:
            mvc_path = os.path.join(MVC_DIR, f"{filename[:-4]}\\{filename.split('_')[0]}\\rig\\mvc.yaml")
            save_path = os.path.join(MVC_DIR, f"{filename[:-4]}\\{filename.split('_')[0]}\\frames")

            with open(mvc_path, 'r') as f:
                content = yaml.safe_load(f)
            
            # change the output directory from the default
            content['controller']['OUTPUT_IMAGE_DIR'] = save_path
            # change camera position
            content['view']['CAMERA_POS'] = [1.06543280, 0.7, 1.398207]
            content['view']['WINDOW_DIMENSIONS'] = [2048,1024]
            
            new_mvc_path = f"{mvc_path[:-4]}_new.yaml"
            with open(new_mvc_path, 'w') as f:
                yaml.safe_dump(content, f)

            user_mvc_cfg_fn = new_mvc_path
            start(user_mvc_cfg_fn)

        except:
            print("Failed! for --> ", filename)
            FAIL_COUNT += 1

    print(f"Failed {FAIL_COUNT} times")
