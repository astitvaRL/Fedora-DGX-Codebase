#!/usr/bin/env python

#  (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import time
import random

import numpy as np
import animated_drawings.streaming.pytelepathy as pytelepathy



"""
This will take the place of most of the current code.
So to start, this will do what current render.py does.
So it will take in the configuration files, create the view,
create the scene, create the controller, etc. 
Actually, why would I move all of that into here?
Really, I just need the client functionality to somehow be 
accessible to the new version of the Animated_Drawing character
so it can query the server as needed.

So I think I just need to write a new AD class that just passes everything to the server.
"""












running = True  # Global flag to control the loop


def handle_sigint(signal, frame):
    print("Interrupt signal received. Exiting the program.")
    global running
    running = False


initialized_characters = np.array([False])


def create_frame_data_request_message():
    global initialized_characters
    message = {
        "TimeUs": 1234567890123,
        "ViewPos": [
            [-0.013721969, 1.02616549, 0.497076154]
        ],
        "JointPos": [[
            -0.07366705,
            0.9294276,
            -0.009044647,
            -0.06748724,
            1.46750665,
            -0.0557479858,
            -0.04018879,
            1.40705776,
            -0.00266361237,
            0.0564217567,
            1.33622122,
            0.18847847,
            -0.0332789421,
            1.35437179,
            0.429159164,
            -0.0964827538,
            1.410176,
            -0.000168800354,
            -0.190067291,
            1.34649181,
            0.1939516,
            -0.09229374,
            1.34989643,
            0.432506561,
            0.006407261,
            0.9040766,
            -0.0106801987,
            -0.00783729553,
            0.48385334,
            -0.01376915,
            -0.0316700935,
            0.06886792,
            -0.07787609,
            -0.15318346,
            0.9028177,
            -0.003920555,
            -0.170153618,
            0.4829812,
            -0.0151433945,
            -0.177796841,
            0.06742144,
            -0.07950401,
        ]],
        "JointPosCharMapping": [
            [0, 0]
        ],
        "CharToReturn": [True]
    }

    character_count = 1
    view_pos_count = len(message['ViewPos'])
    joint_pos_count = len(message['JointPos'])

    # random number of ViewPosCharMapping equal or less than number of characters
    view_pos_char_mapping = []
    for _ in range(random.randint(0, character_count)):
        view_pos_id = random.randint(0, view_pos_count-1)
        character_id = random.randint(0, character_count-1)
        view_pos_char_mapping.append([character_id, view_pos_id])
    message['ViewPosCharMapping'] = view_pos_char_mapping

    # random number of JointPosCharMappings equal or less than number of characters
    joint_pos_char_mapping = []
    for _ in range(random.randint(0, character_count)):
        joint_pos_id = random.randint(0, joint_pos_count-1)
        character_id = random.randint(0, character_count-1)
        joint_pos_char_mapping.append([character_id, joint_pos_id])

        initialized_characters[character_id] = True

    message['JointPosCharMapping'] = joint_pos_char_mapping

    # random number of characters to retun
    message['CharToReturn'] = np.logical_and(np.random.randint(2, size=character_count), initialized_characters).tolist()

    return json.dumps(message)


if __name__ == "__main__":
    # Print program name and instructions
    print("[Telepathy] Mock AD3D Client")
    print("Press Ctrl+C to exit the program at any time.")

    host = "127.0.0.1"
    port = 8888

    # Create the clip data request channel
    clip_req_channel = pytelepathy.ReqChannelConfig()
    clip_req_channel.path = "/clip_data"

    clip_req_config = pytelepathy.WebSocketClientConfig()
    clip_req_config.host = host
    clip_req_config.port = port
    clip_req_config.channel_config = clip_req_channel

    clip_req_client = pytelepathy.WebSocketClient(clip_req_config)
    clip_req_client.start()

    # Create the frame data request channel
    frame_req_channel = pytelepathy.ReqChannelConfig()
    frame_req_channel.path = "/frame_data"

    frame_req_config = pytelepathy.WebSocketClientConfig()
    frame_req_config.host = host
    frame_req_config.port = port
    frame_req_config.channel_config = frame_req_channel

    frame_req_client = pytelepathy.WebSocketClient(frame_req_config)
    frame_req_client.start()

    server_ready = False
    # Keep the program running until SIGINT is received
    try:
        while running:
            time.sleep(1)  # Sleep to reduce CPU usage
            # Send inference request for three times

            if not server_ready and clip_req_client.is_connected():
                request_success = clip_req_client.request(
                    "QueryClipData",
                    1000,
                    lambda response_msg, success: print(
                        f"/clip_data (received: {len(response_msg)} bytes): {response_msg[:16]}..."
                    )
                    if success
                    else print("/clip_data (failed)"),
                )
                server_ready = request_success
                if not request_success:
                    clip_req_client.stop()
                    clip_req_client.start()
            elif server_ready and frame_req_client.is_connected():
                request_success = frame_req_client.request(
                    create_frame_data_request_message(),
                    1000,
                    lambda response_msg, success: print(
                        f"/frame_data (received: {len(response_msg)} bytes): {response_msg[:16]}..."
                    )
                    if success
                    else print("/frame_data (failed)"),
                )
                server_ready = request_success
                if not request_success:
                    frame_req_client.stop()
                    frame_req_client.start()
            else:
                server_ready = False

    except KeyboardInterrupt:
        pass

    print("Client terminated gracefully.")
