# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from animated_drawings.view.view import View
from animated_drawings.model.scene import Scene
from animated_drawings.model.animated_drawing import AnimatedDrawing
import numpy as np
from animated_drawings.config import ViewConfig
import glfw
import OpenGL.GL as GL
import png
import gltfgen
import logging
from pathlib import Path
from typing import List
import functools


class ExportView(View):
    """ Use this view when you want to write character information (verts, active txtrs, etc.) to a file. """

    def __init__(self, cfg: ViewConfig):
        super().__init__(cfg)

        glfw.init()

        self._create_window(*cfg.window_dimensions)

        self.clip_data = {}
        self.frames = []

    def record_clip_level_data(self, scene: Scene, fps: float) -> None:
        """
        Clip level data is constant for the entire render, no need to be changing it per frame.
        Clip level data includes all meshes and txtrs that could be used by the character.
            Each mesh is represented as a list of vertices with 8 attributes:
                -starting x, y, z values [0:3]
                -rgb colors (can ignore these) [3:6]
                -uv texture coords [6:8]
                Note that UV coords are clip_level data, as they don't change per frame.

            Each txtr is an square of RGBA values (or maybe BGRA? TODO: confirm this later)

        Called by ExportController once before rendering frames.
        """

        # to hold meshes and txtrs for all characters within the scene
        clip_level_meshes, clip_level_txtrs = [], []

        for cdx, ad in enumerate(filter(lambda x: isinstance(x, AnimatedDrawing), scene.get_children())):  # for each character in scene
            character_meshes = {}  # holds one character's meshes
            character_txtrs = {}   # holds one character's txtrs

            for mesh_name, ad_mesh in ad.meshes.items():

                character_meshes[mesh_name] = {
                    'vertices': ad_mesh.vertices[:, :8].tolist(),   # (num_verts,8) [x, y, z, r, g, b, u, v]
                    'triangles': ad_mesh.triangles.tolist()  # (num_tris, 3)
                }

                for txtr_name, txtr in ad_mesh.txtr_dict.items():
                    character_txtrs[f'{cdx}__{mesh_name}__{txtr_name}'] = txtr.tolist()

            clip_level_meshes.append(character_meshes)
            clip_level_txtrs.append(character_txtrs)

        self.clip_data['meshes'] = clip_level_meshes
        self.clip_data['txtrs'] = clip_level_txtrs
        self.clip_data['fps'] = fps

    def record_frame_level_data(self, scene: Scene) -> None:
        """
        Frame level data changes every frame.

        For each character, we record the name of the active (visible) mesh and texture.
        These can be used to refer to the proper clip level data.
        We also record the world xyz coordinates for the vertices within the active mesh.
        We compute the vertex normals here (since we haven't had to do so before).
        Finally, we record the vertex render order. This is a list of vertex indices which, when read as chunks of three,
        represents the order in which to render the mesh triangles.

        Called by export ExportController once per frame.
        """

        frame_data = []
        for cdx, ad in enumerate(filter(lambda x: isinstance(x, AnimatedDrawing), scene.get_children())):  # for each character in scene
            assert isinstance(ad, AnimatedDrawing)  # (added to quiet VSCode's static analysis hints)

            character_frame_data = {}

            """ active mesh and txtr names """
            character_frame_data['active_mesh'] = ad.active_mesh_name
            character_frame_data['active_txtr'] = f'{cdx}__{ad.active_mesh_name}__{ad.active_txtr_name}'

            active_mesh = ad.meshes[ad.active_mesh_name]

            """ vertex world positions """
            local_xyz1 = np.hstack((active_mesh.vertices[:, :3], np.ones([active_mesh.vertices.shape[0], 1])))
            world_xyz = (active_mesh.get_world_transform() @ local_xyz1.T).T[:, :3]
            character_frame_data['verts_xyz'] = world_xyz.tolist()  # we only need to update the xyz positions of the vertices. uv doesn't ever change.

            """ uv coords """
            # uv coords don't ever change, so can just get them from the clip_level mesh vertices, like below
            # character_frame_data['verts_uv'] = np.array(self.clip_data['meshes'][cdx][ad.active_mesh_name]['vertices'])[:, 6:8]

            """ vertex normals """
            local_normals_xyz0 = np.hstack((active_mesh.vertices[:, 8:11], np.zeros([active_mesh.vertices.shape[0], 1])))
            world_normals_xyz = (active_mesh.get_world_transform() @ local_normals_xyz0.T).T[:, :3]
            character_frame_data['vert_norm'] = world_normals_xyz.tolist()

            """ indices (vertex render order) """
            character_frame_data['indices'] = active_mesh.indices.tolist()

            frame_data.append(character_frame_data)

        self.frames.append(frame_data)

    def _create_window(self, width: int, height: int) -> None:
        """ This View doesn't use a window, but glfw needs a current context or things get segfaulty. """

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.make_context_current(glfw.create_window(width, height, 'Viewer', None, None))

    def cleanup(self) -> None:
        self.clip_data['frames'] = self.frames
        self.export_gltf("./export/export_dev.glb")

    def export_gltf(self, export_path: str) -> None:
        materials_per_character: List[List[str]] = [list(x.keys()) for x in self.clip_data['txtrs']]
        materials = functools.reduce(lambda a, b: a+b, materials_per_character)

        # Write out the textures, so they can be loaded as pngs
        export_dir_p = Path(export_path).parent
        export_dir_p.mkdir(exist_ok=True)

        logging.info(f"Exporting gltf to {export_dir_p}")

        txtr_dir_p = export_dir_p / "textures"
        txtr_dir_p.mkdir(exist_ok=True)

        for character_txtrs in self.clip_data['txtrs']:
            for name, txtr in character_txtrs.items():
                height = len(txtr)
                if height < 1:
                    continue
                width = len(txtr[0])
                img = [[x for item in row for x in item[:3]] for row in txtr]
                with open(f"{txtr_dir_p}/{name}.png", "wb") as f:
                    w = png.Writer(width, height, greyscale=False)
                    w.write(f, img)

        meshes = []
        for frame in self.clip_data['frames']:
            for cdx, _ in enumerate(frame):  # iterate over characters in frame
                active_mesh_name = frame[cdx]['active_mesh']
                active_mesh = self.clip_data['meshes'][cdx][active_mesh_name]
                active_txtr = frame[cdx]['active_txtr']
                mtlid = materials.index(active_txtr)
                faces = active_mesh["triangles"]
                meshes.append({
                    "n": active_mesh_name,
                    "f": faces,
                    "v": frame[0]['verts_xyz'],
                    "u": [coords[6:] for coords in active_mesh["vertices"]],
                    "e": [mtlid] * len(faces),
                })

        config = {
            "name": "n",
            "faces": "f",
            "vertices": "v",
            "material_attribute": "e",
            "texcoords": {"u": "f32"},
            "textures": [{"embed": f"{txtr_dir_p}/{name}.png"} for name in materials],
            "materials": [
                {
                    "name": name,
                    "base_texture": {
                        "index": i,
                        "texcoord": 0,
                    }
                } for [i, name] in enumerate(materials)
            ],
            "fps": self.clip_data["fps"],
            "insert_vanishing_frames": True,
            "reverse": False,
            "invert_tets": False,
            "output": export_path
        }

        gltfgen.export(meshes, config)

        # cleanup texture files
        for item in txtr_dir_p.glob("./*"):
            item.unlink()
        txtr_dir_p.rmdir()
