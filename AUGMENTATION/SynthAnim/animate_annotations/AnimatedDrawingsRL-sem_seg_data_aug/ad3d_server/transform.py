# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations  # so we can refer to class Type inside class

from ad3d_server.vectors import Vectors
from ad3d_server.quaternions import Quaternions

import numpy as np
import numpy.typing as npt
import logging
from typing import Union, Optional, List
from collections import Iterable


class Transform():
    """Base class from which all other scene objects descend"""

    def __init__(self,
                 parent: Optional[Transform] = None,
                 name: Optional[str] = None,
                 children: List[Transform] = [],
                 offset: Union[npt.NDArray[np.float32], Vectors, None] = None,
                 rotation: Optional[Quaternions] = None,
                 **kwargs
                 ) -> None:

        super().__init__(**kwargs)

        self._parent: Optional[Transform] = parent

        self.name: Optional[str] = name

        self._translate_m: npt.NDArray[np.float32] = np.identity(4, dtype=np.float32)
        self._rotate_m: npt.NDArray[np.float32] = np.identity(4, dtype=np.float32)
        self._scale_m: npt.NDArray[np.float32] = np.identity(4, dtype=np.float32)

        self._local_transform: npt.NDArray[np.float32] = np.identity(4, dtype=np.float32)
        self._world_transform: npt.NDArray[np.float32] = np.identity(4, dtype=np.float32)

        if offset is not None:
            self.offset(offset)

        if rotation is not None:
            self.set_rotation(rotation)

        self._dirty_bit: bool = True            # are my world/local transforms stale?
        self._children_dirty_bit: bool = False  # did I (not) update my children's world transforms after changing my own?
        self.update_transforms(recurse_on_children=False)

        self.is_visible = True

        self._children: List[Transform] = []
        for child in children:
            self.add_child(child)

    def update_transforms(self, parent_children_dirty_bit: bool = False, recurse_on_children: bool = True, update_ancestors: bool = False) -> None:
        """
        Called by other classes to update transforms.  If update_ancestors is True,
        this will ensure all parents of current transform have up-to-date local and world matrices before computing
        it's own local and world matrices. If recurse_on_children is True, will update the local and world matrices
        of all children of the current Transform, then reset _children_dirty_bit.
        """
        if update_ancestors:

            # populate ancestral_line until no more parents remain
            ancestor: Optional[Transform] = self.get_parent()
            ancestral_line: List[Transform] = []
            is_ancestor_dirty = False
            while ancestor is not None:
                ancestral_line.append(ancestor)
                ancestor = ancestor.get_parent()

            # proceed from first ancestor downward, keeping track of whether any have been dirty and updating as needed
            for ancestor in ancestral_line[::-1]:
                is_ancestor_dirty |= ancestor._dirty_bit
                ancestor._update_own_transforms(is_ancestor_dirty)

        self._update_own_transforms()

        if recurse_on_children:
            for c in self.get_children():
                c.update_transforms(self._children_dirty_bit | parent_children_dirty_bit)
            self._children_dirty_bit = False

    def _update_own_transforms(self, force_world_transform_update=True) -> None:
        """ updates local transform if dirty bit is set. updates world transform if dirty bit is set or force_world_transform_update is true. """
        if self._dirty_bit:
            self._compute_local_transform()
        if self._dirty_bit | force_world_transform_update:
            self._compute_world_transform()
        self._dirty_bit = False

    def _compute_local_transform(self) -> None:
        self._local_transform = self._translate_m @ self._rotate_m @ self._scale_m

    def _compute_world_transform(self) -> None:
        self._world_transform = self._local_transform
        if self._parent:
            self._world_transform = self._parent._world_transform @ self._world_transform

    def update_and_get_world_transform(self) -> npt.NDArray[np.float32]:
        """ Ensures this transforms world matrix is up-to-date and returns it. """
        self.update_transforms(recurse_on_children=False, update_ancestors=True)
        return self.get_world_transform()

    def get_world_transform(self) -> npt.NDArray[np.float32]:
        return np.copy(self._world_transform)

    def set_scale(self, scale: Union[Iterable, float]) -> None:
        if isinstance(scale, Iterable):
            assert len(scale) == 3, 'scale is iterable but not len==3'
        self._scale_m[:-1, :-1] = scale * np.identity(3, dtype=np.float32)
        self._set_dirty_bits()

    def set_position(self, pos: Union[npt.NDArray[np.float32], Vectors]) -> None:
        """ Set the absolute values of the translational elements of transform """
        if isinstance(pos, Vectors):
            pos = pos.vs

        if pos.shape == (1, 3):
            pos = np.squeeze(pos)
        elif pos.shape == (3,):
            pass
        else:
            msg = f'bad vector dim passed to set_position. Found: {pos.shape}'
            logging.critical(msg)
            assert False, msg

        self._translate_m[:-1, -1] = pos
        self._set_dirty_bits()

    def get_local_position(self) -> npt.NDArray[np.float32]:
        """ Ensure local transform is up-to-date and return local xyz coordinates """
        if self._dirty_bit:
            self._compute_local_transform()
        return np.copy(self._local_transform[:-1, -1])

    def update_and_get_world_position(self) -> npt.NDArray[np.float32]:
        """ Ensures this transforms world matrix is up-to-date and returns world cartesian coordinates. """
        self.update_transforms(recurse_on_children=False, update_ancestors=True)
        return self.get_world_position()

    def get_world_position(self) -> npt.NDArray[np.float32]:
        return np.copy(self._world_transform[:-1, -1])

    def offset(self, pos: Union[npt.NDArray[np.float32], Vectors]) -> None:
        """ Translational offset by the specified amount """

        if isinstance(pos, Vectors):
            pos = pos.vs[0]
        assert isinstance(pos, np.ndarray)

        self.set_position(self._translate_m[:-1, -1] + pos)

    def look_at_spherical_coordinates(self, v_angle: float, h_angle: float) -> None:
        direction = np.array([np.cos(v_angle) * np.sin(h_angle), np.sin(v_angle), np.cos(v_angle) * np.cos(h_angle)])
        self.look_at(direction)

    def look_at(self, fwd_: Union[npt.NDArray[np.float32], Vectors, None]) -> None:
        """Given a forward vector, rotate the transform to face that position"""
        if fwd_ is None:
            fwd_ = Vectors(self.update_and_get_world_position())
        elif isinstance(fwd_, np.ndarray):
            fwd_ = Vectors(fwd_)
        fwd: Vectors = fwd_.copy()  # norming will change the vector

        if fwd.vs.shape != (1, 3):
            msg = f'look_at fwd_ vector must have shape [1,3]. Found: {fwd.vs.shape}'
            logging.critical(msg)
            assert False, msg

        tmp: Vectors = Vectors([0.0, 1.0, 0.0])

        # if fwd and tmp are same vector, modify tmp to avoid collapse
        if np.isclose(fwd.vs, tmp.vs).all() or np.isclose(fwd.vs, -tmp.vs).all():
            tmp.vs[0] += 0.001

        right: Vectors = tmp.cross(fwd)
        up: Vectors = fwd.cross(right)

        fwd.norm()
        right.norm()
        up.norm()

        rotate_m = np.identity(4, dtype=np.float32)
        rotate_m[:-1, 0] = np.squeeze(right.vs)
        rotate_m[:-1, 1] = np.squeeze(up.vs)
        rotate_m[:-1, 2] = np.squeeze(fwd.vs)

        self._rotate_m = rotate_m
        self._set_dirty_bits()

    def set_rotation(self, q: Quaternions) -> None:
        if q.qs.shape != (1, 4):
            msg = f'set_rotate q must have dimension (1, 4). Found: {q.qs.shape}'
            logging.critical(msg)
            assert False, msg
        self._rotate_m = q.to_rotation_matrix()
        self._set_dirty_bits()

    def rotation_offset(self, q: Quaternions) -> None:
        if q.qs.shape != (1, 4):
            msg = f'set_rotate q must have dimension (1, 4). Found: {q.qs.shape}'
            logging.critical(msg)
            assert False, msg
        self._rotate_m = (q * Quaternions.from_rotation_matrix(self._rotate_m)).to_rotation_matrix()
        self._set_dirty_bits()

    def add_child(self, child: Transform) -> None:
        self._children.append(child)
        child.set_parent(self)
        self._children_dirty_bit = True

    def remove_child(self, child: Transform) -> None:
        try:
            self._children.remove(child)
        except ValueError as e:
            logging.info(f'Attempted to remove non-existant child: {self} | {child} : {e}')

    def get_children(self) -> List[Transform]:
        return self._children

    def set_parent(self, parent: Transform) -> None:
        self._parent = parent
        self._set_dirty_bits()

    def get_parent(self) -> Optional[Transform]:
        return self._parent

    def _set_dirty_bits(self) -> None:
        self._dirty_bit = True
        self._children_dirty_bit = True

    def get_transform_by_name(self, name: str) -> Transform:

        transform = self._get_transform_by_name(name)

        if transform:
            return transform

        raise ValueError(f'Could not find transform named {name}')

    def _get_transform_by_name(self, name: str) -> Optional[Transform]:
        """ Search self and children for transform with matching name. Return it if found, None otherwise. """

        if self.name == name:
            return self  # we are match

        # recurse to check if a child is match
        for child in self.get_children():
            try:
                return child.get_transform_by_name(name)
            except ValueError:
                pass  # no match in this child or its children

        return None  # no match found

    def add_transorm_widget(self) -> None:
        self.add_child(TransformWidget())

    def draw(self, recurse: bool = True, **kwargs) -> None:
        """ Draw this transform and recurse on children """

        if not self.is_visible:
            return

        self._draw(**kwargs)

        if not recurse:
            return

        for child in self.get_children():
            child.draw(**kwargs)

    def _draw(self, **kwargs) -> None:
        """Transforms default to not being drawn. Subclasses must implement how they appear"""


class TransformWidget(Transform):
    def __init__(self, shader_name: str = 'color_shader'):

        super().__init__()

        self.points: npt.NDArray[np.float32] = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.3, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.3, 0.0, 0.0, 1.0],
        ], np.float32)

        self.shader_name: str = shader_name
