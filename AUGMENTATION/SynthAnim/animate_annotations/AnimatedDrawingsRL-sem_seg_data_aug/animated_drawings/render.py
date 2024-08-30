# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import sys
import asyncio


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

    # user-specified mvc configuration filepath. Can be absolute, relative to cwd, or relative to ${AD_ROOT_DIR}
    # user_mvc_cfg_fn = sys.argv[1]
    user_mvc_cfg_fn = "example_images\\3\\rig\\mvc.yaml"


    start(user_mvc_cfg_fn)
