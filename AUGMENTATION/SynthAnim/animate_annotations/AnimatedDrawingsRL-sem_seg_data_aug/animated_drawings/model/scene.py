# Copyright (c) Meta Platforms, Inc. and affiliates.

from animated_drawings.model.transform import Transform
from animated_drawings.model.camera import Camera
from animated_drawings.model.time_manager import TimeManager
from animated_drawings.config import SceneConfig
from animated_drawings.model.floor import Floor
from animated_drawings.model.animated_drawing_client import AnimatedDrawing

from animated_drawings.model.motion_source import MotionSource

class Scene(Transform, TimeManager):
    """
    The scene is the singular 'world' object.
    It contains all objects that need to be drawn.
    It keeps track of global time.
    """

    def __init__(self, cfg: SceneConfig, cam: Camera) -> None:
        """ Takes in the scene dictionary from an mvc config file and prepares the scene. """
        super().__init__()

        # add floor
        self.add_child(Floor())

        # Add the Animated Drawings
        for char_cfg, retarget_cfg, motion_cfg in cfg.animated_characters:

            motion_source = MotionSource.create_motion_source(motion_cfg)
            self.add_child(motion_source.get_visualization_transform())

            ad = AnimatedDrawing(char_cfg, retarget_cfg, viewer=cam)
            ad.set_motion_source(motion_source)
            self.add_child(ad)

            # # # add motion source viz widget to scene if config specifies it
            # if cfg.add_ad_retarget_bvh:
            #     self.add_child(ad.retargeter.motion_source.get_visualization_transform())

    def progress_time(self, delta_t: float) -> None:
        """
        Entry point called to update time in the scene by delta_t seconds.
        Because animatable object within the scene may have their own individual timelines,
        we recurvisely go through objects in the scene and call tick() on each TimeManager.
        """
        self._progress_time(self, delta_t)

    def _progress_time(self, t: Transform, delta_t: float) -> None:
        """ Recursively calls tick() on all TimeManager objects. """

        if isinstance(t, TimeManager):
            t.tick(delta_t)

        for c in t.get_children():
            self._progress_time(c, delta_t)
