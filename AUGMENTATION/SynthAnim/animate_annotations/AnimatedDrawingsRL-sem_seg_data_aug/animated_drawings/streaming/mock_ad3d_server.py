#!/usr/bin/env python

#  (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import signal
import sys
import time

import animated_drawings.streaming.pytelepathy as pytelepathy

running = True  # Add this global flag to control the loop


def create_clip_level_data() -> str:
    message = {"Characters": [{"Name": "test character", "Meshes": [], "Textures": []}]}
    return json.dumps(message)


def create_frame_level_data() -> str:
    message = {"Characters": [], "Timestamp": 12345678}
    return json.dumps(message)


def handle_sigint(sig, frame):
    print("Interrupt signal received. Exiting the program.")
    global running
    running = False


def clip_res_channel_callback(request_msg):
    print(
        f"/clip_data (received_request: {len(request_msg)} bytes): {request_msg[:16]}..."
    )
    return create_clip_level_data()


def frame_res_channel_callback(request_msg):
    print(
        f"/frame_data (received_request: {len(request_msg)} bytes): {request_msg[:16]}..."
    )
    return create_frame_level_data()


if __name__ == "__main__":
    # Print program name and instructions
    print("[Telepathy] Mock Dyno Server")
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

    # ResChannel configuration
    clip_res_channel = pytelepathy.ResChannelConfig()
    clip_res_channel.path = "/clip_data"
    clip_res_channel.callback = clip_res_channel_callback
    config.res_channels.append(clip_res_channel)

    frame_res_channel = pytelepathy.ResChannelConfig()
    frame_res_channel.path = "/frame_data"
    frame_res_channel.callback = frame_res_channel_callback
    config.res_channels.append(frame_res_channel)

    server = pytelepathy.WebSocketServer(config)
    server.start()

    # Keep the program running until SIGINT is received
    try:
        while running:
            time.sleep(1)  # Sleep to reduce CPU usage
    except KeyboardInterrupt:
        pass

    print("Client terminated gracefully.")
