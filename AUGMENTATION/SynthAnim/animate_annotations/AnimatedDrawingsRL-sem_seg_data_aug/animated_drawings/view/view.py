# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations
from abc import abstractmethod
from typing import Tuple
from animated_drawings.config import ViewConfig
from animated_drawings.model.camera import Camera


class View:
    """
    Base View class which all other Views must be derived.
    Views are responsible for controlling what is and isn't visible to them.
    Views are responsible for initiating the 'draw' methods for each object which they want to render.
    """

    def __init__(self, cfg: ViewConfig):
        self.cfg: ViewConfig = cfg

        self.camera: Camera = Camera(cfg.camera_pos, cfg.camera_fwd)

    @abstractmethod
    def render(self, scene) -> None:  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
        """ Called by the controller to render the scene. """

    @abstractmethod
    def clear_window(self) -> None:
        """ Clear output from previous render loop. """

    @abstractmethod
    def cleanup(self) -> None:
        """ Cleanup after render loop is finished. """

    @abstractmethod
    def get_framebuffer_size(self) -> Tuple[int, int]:
        """ Return (width, height) of framebuffer. """

    @staticmethod
    def create_view(view_cfg: ViewConfig) -> View:
        """ Takes in a view dictionary from mvc config file and returns the appropriate view. """
        # create view
        if view_cfg.view_type == 'mesa':
            from animated_drawings.view.mesa_view import MesaView
            return MesaView(view_cfg)
        elif view_cfg.view_type == 'window':
            from animated_drawings.view.window_view import WindowView
            return WindowView(view_cfg)
        elif view_cfg.view_type == 'export':
            from animated_drawings.view.export_view import ExportView
            return ExportView(view_cfg)
        elif view_cfg.view_type == 'streaming':
            from animated_drawings.view.streaming_view import StreamingView
            return StreamingView(view_cfg)
