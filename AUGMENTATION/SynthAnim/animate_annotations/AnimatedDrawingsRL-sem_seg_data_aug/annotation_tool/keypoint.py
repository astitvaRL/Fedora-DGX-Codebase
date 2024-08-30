
from __future__ import annotations
from typing import Tuple, Union


class Keypoint():
    def __init__(self, xy: Tuple[float, float], c: float, name: str, parent: Union[Keypoint, str]):
        self.x = xy[0]
        self.y = xy[1]
        self.c = c
        self.name = name
        self.parent = parent
        self.children = []
