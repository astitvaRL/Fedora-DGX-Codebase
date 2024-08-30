#!/usr/bin/env python

#  (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import time

import numpy as np
import animated_drawings.streaming.pytelepathy as pytelepathy
from pathlib import Path

running = True  # Global flag to control the loop

image_fn = str(Path('./metagen_example_sticker.png').resolve())  # example image being sent to server


def handle_sigint(signal, frame):
    print("Interrupt signal received. Exiting the program.")
    global running
    running = False


initialized_characters = np.array([False])

if __name__ == "__main__":
    # Print program name and instructions
    print("[Telepathy] Mock AD3D Client")
    print("Press Ctrl+C to exit the program at any time.")

    host = "127.0.0.1"
    port = 8884

    # Create the clip data request channel
    annotation_req_channel = pytelepathy.ReqChannelConfig()
    annotation_req_channel.path = "/annotation_data"

    annotation_req_config = pytelepathy.WebSocketClientConfig()
    annotation_req_config.host = host
    annotation_req_config.port = port
    annotation_req_config.channel_config = annotation_req_channel

    annotation_req_client = pytelepathy.WebSocketClient(annotation_req_config)
    annotation_req_client.start()

    server_ready = False

    # Keep the program running until SIGINT is received
    try:
        while running:
            time.sleep(1)  # Sleep to reduce CPU usage
            # Send inference request for three times
            if not server_ready and annotation_req_client.is_connected():
                request_success = annotation_req_client.request(
                    image_fn,
                    1000,
                    lambda response_msg, success: print(
                        f"/annotation_data (received: {len(response_msg)} bytes): {response_msg}..."
                    )
                    if success
                    else print("/annotation_data (failed)"),
                )
                server_ready = request_success
                if not request_success:
                    annotation_req_client.stop()
                    annotation_req_client.start()
            else:
                server_ready = False

    except KeyboardInterrupt:
        pass

    print("Client terminated gracefully.")
