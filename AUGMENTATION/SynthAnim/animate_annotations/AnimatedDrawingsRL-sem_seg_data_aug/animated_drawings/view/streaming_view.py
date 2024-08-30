# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from animated_drawings.view.view import View
from animated_drawings.model.scene import Scene
from animated_drawings.model.animated_drawing import AnimatedDrawing
from typing import List
import numpy as np
from animated_drawings.config import ViewConfig
import glfw
import OpenGL.GL as GL
import png
import io
import base64
import json
import animated_drawings.streaming.pytelepathy as pytelepathy


class StreamingView(View):
    """
    Use this view to compile message used for websocket streaming to the Unity demo.
    It compiles two types of data:
    `clip_data` is used for getting global information about the clip. It is usually large and should only be retrived when necessary.
    `frame_data` is used for getting information for each frame, that's necessary for animating the character.
    """

    def __init__(self, cfg: ViewConfig):
        super().__init__(cfg)
        glfw.init()
        self._create_window(*cfg.window_dimensions)

    def _create_window(self, width: int, height: int) -> None:
        """ This View doesn't use a window, but glfw needs a current context or things get segfaulty. """

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.make_context_current(glfw.create_window(width, height, 'Viewer', None, None))

    def get_clip_level_data(self, scene: Scene) -> str:
        clip_data = {}
        clip_data["Characters"] = []

        for cdx, ad in enumerate(filter(lambda x: isinstance(x, AnimatedDrawing), scene.get_children())):  # for each character in scene
            character_data = {}
            character_data['Name'] = f'{cdx}'
            character_data['Meshes'] = []
            character_data['Textures'] = []

            for mesh_name, ad_mesh in ad.meshes.items():
                mesh_data = {}
                mesh_data['Name'] = mesh_name
                mesh_data['Vertices'] = ad_mesh.vertices[:ad_mesh.front_vertex_count, :3].flatten().tolist()
                mesh_data['UVs'] = ad_mesh.vertices[:ad_mesh.front_vertex_count, 6:8].flatten().tolist()
                mesh_data['Triangles'] = ad_mesh.triangles[ad_mesh.front_triangles_start:ad_mesh.front_triangles_end].flatten().tolist()

                character_data['Meshes'].append(mesh_data)

                for txtr_name, txtr in ad_mesh.txtr_dict.items():
                    texture_data = {}
                    texture_data["Name"] = f'{cdx}__{mesh_name}__{txtr_name}'
                    # Construct texture png
                    height = len(txtr)
                    width = len(txtr)
                    img = [[x for item in row for x in item[:3]] for row in txtr]
                    buffer = io.BytesIO()
                    w = png.Writer(width, height, greyscale=False)
                    w.write(buffer, img)
                    base64_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

                    texture_data["Data"] = base64_encoded
                    character_data['Textures'].append(texture_data)
            clip_data["Characters"].append(character_data)

        return json.dumps(clip_data)

    def get_frame_level_data(self, scene: Scene, chars_to_return: List[bool]) -> None:
        frame_data = {}
        frame_data["Characters"] = []
        # TODO set correct timestamp
        frame_data["Timestamp"] = pytelepathy.get_time_ns()

        animated_drawings_in_scene: List[AnimatedDrawing] = [child for child in scene.get_children() if isinstance(child, AnimatedDrawing)]
        for cdx, ad in enumerate(animated_drawings_in_scene):  # for each character in scene

            # if this character doesn't need to be streamed back, append an empty dictionary in its place
            if not chars_to_return[cdx]:
                frame_data["Characters"].append({})
                continue

            # otherwise, gather the appropriate data to return

            character_frame_data = {}

            """ active mesh and txtr names """
            character_frame_data['ActiveMesh'] = ad.active_mesh_name
            character_frame_data['ActiveTexture'] = f'{cdx}__{ad.active_mesh_name}__{ad.active_txtr_name}'

            active_mesh = ad.meshes[ad.active_mesh_name]

            """ vertex world positions """
            local_xyz1 = np.hstack((active_mesh.vertices[:active_mesh.front_vertex_count, :3], np.ones([active_mesh.front_vertex_count, 1])))
            world_xyz = (active_mesh.get_world_transform() @ local_xyz1.T).T[:, :3]
            character_frame_data['Vertices'] = world_xyz.flatten().tolist()  # we only need to update the xyz positions of the vertices. uv doesn't ever change.

            """ uv coords """
            # uv coords don't ever change, so can just get them from the clip_level mesh vertices, like below
            # character_frame_data['verts_uv'] = np.array(self.clip_data['meshes'][cdx][ad.active_mesh_name]['vertices'])[:, -2:]

            """ vertex normals """
            local_normals_xyz0 = np.hstack((active_mesh.vertices[:active_mesh.front_vertex_count, 8:11], np.zeros([active_mesh.front_vertex_count, 1])))
            world_normals_xyz = (active_mesh.get_world_transform() @ local_normals_xyz0.T).T[:, :3]
            character_frame_data['Normals'] = world_normals_xyz.flatten().tolist()

            """ indices (vertex render order) """
            character_frame_data['Indices'] = active_mesh.indices[3*active_mesh.front_triangles_start:3*active_mesh.front_triangles_end].tolist()

            frame_data["Characters"].append(character_frame_data)

        return json.dumps(frame_data)
