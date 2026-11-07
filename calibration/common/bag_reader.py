"""Minimal rosbag2 reader, built on rosbag2_py (bundled with any ROS2
install -- no extra pip packages needed on a machine that already has ROS2,
which is the common case for this project). Deliberately small: just
"iterate (topic, msg, timestamp_ns) in time order, optionally filtered to
one topic" -- everything else lives in the calibration scripts.

If you're running this on a machine WITHOUT ROS2 (e.g. your own laptop,
not the Pi), install the `rosbags` package instead (`pip install rosbags`)
and use its `rosbags.rosbag2.Reader` + `rosbags.serde.deserialize_cdr` in
place of this module -- same (topic, msg, timestamp) shape, pure Python,
no ROS install required. Not wired up here because it wasn't possible to
pip-install it on the Pi this was developed on (no python3-venv, no
passwordless sudo) -- this is the one part of this repo that was designed
but not run end-to-end off-ROS. If you hit this limitation, that's the
first thing to fix.
"""
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_bag(bag_path, topics=None):
    """Yields (topic_name, msg, timestamp_ns) in the order stored (which
    rosbag2 guarantees is arrival/recording order, not necessarily sorted
    across topics -- sort downstream if you need strict global ordering).
    `topics`: None for everything, or an iterable of topic names to keep.
    """
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = set(topics) if topics else None

    while reader.has_next():
        topic_name, data, t = reader.read_next()
        if wanted is not None and topic_name not in wanted:
            continue
        msg_type = get_message(type_map[topic_name])
        msg = deserialize_message(data, msg_type)
        yield topic_name, msg, t


def image_msg_to_numpy(msg):
    """sensor_msgs/Image -> HxWxC (or HxW) numpy array. Handles the
    encodings this project's own camera nodes actually publish (rgb8/
    bgr8/mono8) -- extend if you add others."""
    import numpy as np
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('rgb8', 'bgr8'):
        img = arr.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            img = img[:, :, ::-1]  # -> bgr8, OpenCV's native order
        return img
    if msg.encoding == 'mono8':
        return arr.reshape(msg.height, msg.width)
    raise ValueError(f'unhandled image encoding: {msg.encoding!r} -- add it to image_msg_to_numpy')


def pointcloud2_to_xyz(msg):
    """sensor_msgs/PointCloud2 -> Nx3 float32 array of (x,y,z), ignoring
    any other fields (intensity etc.). Reads x/y/z at their declared
    per-point byte offsets with the message's real point_step as the
    struct's itemsize -- correct regardless of what other fields (or
    padding) sit between them, unlike naively slicing the first 12 bytes
    of each point."""
    import numpy as np
    offsets = {f.name: f.offset for f in msg.fields if f.name in ('x', 'y', 'z')}
    missing = {'x', 'y', 'z'} - offsets.keys()
    if missing:
        raise ValueError(f'PointCloud2 is missing field(s): {missing}')
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [np.float32, np.float32, np.float32],
        'offsets': [offsets['x'], offsets['y'], offsets['z']],
        'itemsize': msg.point_step,
    })
    n_points = len(msg.data) // msg.point_step
    structured = np.frombuffer(msg.data, dtype=dtype, count=n_points)
    return np.column_stack([structured['x'], structured['y'], structured['z']])
