#!/usr/bin/env python3
"""Live MJPEG preview of the raw RGB topic -- for checking framing and
adjusting the lens focus ring by eye, which the main fused point-cloud
viewer (http://<pi-ip>:8080/) is too low-detail/low-rate for. Runs its own
tiny HTTP server on a separate port so it doesn't interfere with the main
streamer.

Usage: rgb_live_preview.py [--topic /landfill/rgb/image_raw] [--port 8081]
Then open http://<pi-ip>:8081/ in a browser on the same network and leave
it open while you turn the focus ring -- it updates continuously, not a
single snapshot.
"""
import argparse
import io
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

latest_jpeg = {'data': None, 'seq': 0}
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep stdout clean

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''<!doctype html><html><body style="margin:0;background:#111">
<img src="/stream.mjpeg" style="width:100%;height:auto;display:block">
</body></html>''')
            return
        if self.path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            last_seq = -1
            try:
                while True:
                    with lock:
                        data, seq = latest_jpeg['data'], latest_jpeg['seq']
                    if data is not None and seq != last_seq:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(data)
                        self.wfile.write(b'\r\n')
                        last_seq = seq
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_response(404)
        self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/landfill/rgb/image_raw')
    ap.add_argument('--port', type=int, default=8081)
    args = ap.parse_args()

    rclpy.init()
    node = Node('rgb_live_preview')

    def cb(msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        bgr = arr[:, :, ::-1] if msg.encoding == 'rgb8' else arr
        ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with lock:
                latest_jpeg['data'] = jpeg.tobytes()
                latest_jpeg['seq'] += 1

    node.create_subscription(Image, args.topic, cb, qos_profile_sensor_data)

    server = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f'Serving live RGB preview on http://{ip}:{args.port}/ (topic: {args.topic})')
    print('Open that in a browser and leave it open while adjusting focus. Ctrl-C to stop.')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
