import json
import signal
import sys
import time
import numpy as np
from pathlib import Path
import pickle
import tempfile


import animated_drawings.streaming.pytelepathy as pytelepathy
from ad3d_server.animated_drawing import ServerAnimatedDrawing


running = True  # Add this global flag to control the loop

animated_drawing = None


def create_clip_level_data(request_msg) -> str:
    """ When queried and passed config file path names, will create the AnimatedDrawing here on server and stream universe of its meshes back """
    # parse and validate input from clinet
    char_cfg_fn = request_msg
    assert Path(char_cfg_fn).exists(), f'char_cfg_fn DNE: {char_cfg_fn}'

    # create animated_drawing and store in global namespace
    global animated_drawing

    temp_folder_path = tempfile.gettempdir()
    cached_file_loc = Path(temp_folder_path) / f"{Path(char_cfg_fn).name}.pickle"
    # first, check if cached version of character exists
    if not cached_file_loc.exists():
        # create the animated drawing
        animated_drawing = ServerAnimatedDrawing(char_cfg_fn)
        # then save it
        with open(f'{cached_file_loc.absolute()}', 'wb') as f:
            pickle.dump(animated_drawing, f)
    else:
        # load the animated drawing from the file
        with open(f'{cached_file_loc}', 'rb') as f:
            animated_drawing = pickle.load(f)

    # get the animated_drawing's mesh universe and stream back
    meshes = animated_drawing.get_meshes_for_client()

    return json.dumps(meshes).encode('utf-8')


def create_frame_level_data(request_msg) -> str:

    # parse and validate input from clinet
    request_json = json.loads(request_msg)
    viewer_position = request_json["viewer_position"]
    character_position = request_json["character_position"]
    ms_joint_names = request_json["ms_joint_names"]
    ms_joint_positions = request_json["ms_joint_positions"]
    ms_forward_vector = request_json["ms_forward_vector"]
    ms_root_offset = request_json["ms_root_offset"]
    try:
        viewer_position = np.array(viewer_position)
        character_position = np.array(character_position)
        ms_joint_names = ms_joint_names
        ms_joint_positions = np.array(ms_joint_positions).reshape(-1, 3)
        ms_forward_vector = np.array(ms_forward_vector)
        ms_root_offset = np.array(ms_root_offset)
    except Exception as e:
        assert False, f'Error validating frame_level_data request: {e}'

    # ensure everything's properly initiatlized
    assert animated_drawing is not None, 'server_animated_drawing has not been set'
    # update the viewer positon
    animated_drawing.set_viewer_position(viewer_position)
    # update character position
    animated_drawing.set_position(character_position)
    # update the motion source joint positions
    animated_drawing.set_motion_source_joint_positions(ms_joint_names, ms_joint_positions)
    # update the motion source forward vector
    animated_drawing.set_motion_source_forward_vector(ms_forward_vector)
    # update motion source root offset
    animated_drawing.set_motion_source_root_offset(ms_root_offset)

    # perform the retargeting
    animated_drawing.update()

    # get the plane angle for client
    theta = animated_drawing.get_plane_rotation_for_client()
    # get character's position for client
    position = animated_drawing.get_animated_drawing_translation_for_client()
    # get updated xy coordinates of mesh vertices
    meshname_updated_xy_list = animated_drawing.get_mesh_vertex_xy_updates_for_client()
    # get updated active texture's for meshes
    meshname_active_texturename_list = animated_drawing.get_mesh_active_texture_names_for_client()
    # get updated mesh render orders
    meshes_renderorders = animated_drawing.get_meshes_renderorders_for_client()  # a list of (mesh_name, render_indices) in the order in which they should be rendered.

    message = {
        "theta": float(theta),
        "position": position.flatten().tolist(),
        "meshname_updated_xy_list": meshname_updated_xy_list,
        "meshname_active_texturename_list": meshname_active_texturename_list,
        "meshes_renderorders": meshes_renderorders
    }

    return json.dumps(message).encode('utf-8')


def register_motion_source(request_msg): 
    animated_drawing.register_motion_source(json.loads(request_msg))
    return "new motion source registered"


def handle_sigint(sig, frame):
    print("Interrupt signal received. Exiting the program.")
    global running
    running = False


def clip_res_channel_callback(request_msg):
    print(request_msg)
    print(
        f"/clip_data (received_request: {len(request_msg)} bytes): {request_msg[:16]}..."
    )
    return create_clip_level_data(request_msg)


def frame_res_channel_callback(request_msg):
    print(
        f"/frame_data (received_request: {len(request_msg)} bytes): {request_msg[:16]}..."
    )
    return create_frame_level_data(request_msg)


def register_motion_source_res_channel_callback(request_msg):
    print(
        f"/register_motion_source_data (received_request: {len(request_msg)} bytes): {request_msg[:16]}..."
    )
    return register_motion_source(request_msg)


if __name__ == "__main__":
    # Print program name and instructions
    print("[Telepathy] AD3D Server")
    print("Press Ctrl+C to exit the program at any time.")

    # Register signal and signal handler
    signal.signal(signal.SIGINT, handle_sigint)

    host = "::"  # Default host (any IPv4 and IPv6)
    port = 8888  # Default port

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
    clip_res_channel = pytelepathy.ResChannelConfig()
    clip_res_channel.path = "/clip_data"
    clip_res_channel.callback = clip_res_channel_callback
    config.res_channels.append(clip_res_channel)

    frame_res_channel = pytelepathy.ResChannelConfig()
    frame_res_channel.path = "/frame_data"
    frame_res_channel.callback = frame_res_channel_callback
    config.res_channels.append(frame_res_channel)

    register_motion_source_res_channel = pytelepathy.ResChannelConfig()
    register_motion_source_res_channel.path = "/register_motion_source_data"
    register_motion_source_res_channel.callback = register_motion_source_res_channel_callback
    config.res_channels.append(register_motion_source_res_channel)

    server = pytelepathy.WebSocketServer(config)
    server.start()

    # Keep the program running until SIGINT is received
    try:
        while running:
            time.sleep(1)  # Sleep to reduce CPU usage
    except KeyboardInterrupt:
        pass

    print("Client terminated gracefully.")
