import os
import json
import glob
import argparse

import numpy as np
import bagpy
from bagpy import bagreader
from tqdm import tqdm
import cv2

def timestamp_float(sec, nsec):
    return sec + nsec / 1e9

def bag_to_npy(bag_path, output_pth, event_topic, image_topic):
    if not os.path.exists(output_pth):
        os.makedirs(output_pth, exist_ok=True)

    bag = bagreader(bag_path)

    xs, ys, ts, ps = [], [], [], []
    image_list, image_ts_list = [], []

    print("🔄 Reading events...")
    for topic, msg, t in tqdm(bag.bag.read_messages(topics=[event_topic]), desc='🧠 Events'):
        for e in msg.events:
            timestamp = timestamp_float(e.ts.secs, e.ts.nsecs)
            xs.append(e.x)
            ys.append(e.y)
            ps.append(1 if e.polarity else 0)
            ts.append(timestamp)

    print("🖼️ Reading images...")
    for topic, msg, t in tqdm(bag.bag.read_messages(topics=[image_topic]), desc='📷 Images'):
        timestamp = timestamp_float(msg.header.stamp.secs, msg.header.stamp.nsecs)
        image_ts_list.append(timestamp)

        # 将ROS图像消息转为OpenCV格式
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ Warning: failed to decode image at {timestamp}")
            continue

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        image_list.append(img)

    events_ts = np.array(ts, dtype=np.float64)
    events_xy = np.array([xs, ys], dtype=np.int32).T
    events_p = np.array(ps, dtype=np.int8)

    images = np.stack(image_list, axis=0)
    images_ts = np.array(image_ts_list, dtype=np.float64).reshape(-1, 1)

    min_ts = min(events_ts.min(), images_ts.min())
    events_ts -= min_ts
    images_ts -= min_ts

    image_event_indices = np.searchsorted(events_ts, images_ts.reshape(-1), 'right') - 1
    image_event_indices = np.clip(image_event_indices, 0, len(events_ts) - 1)

    np.save(os.path.join(output_pth, 'events_ts.npy'), events_ts)
    np.save(os.path.join(output_pth, 'events_xy.npy'), events_xy)
    np.save(os.path.join(output_pth, 'events_p.npy'), events_p)
    np.save(os.path.join(output_pth, 'images.npy'), images)
    np.save(os.path.join(output_pth, 'images_ts.npy'), images_ts)
    np.save(os.path.join(output_pth, 'image_event_indices.npy'), image_event_indices)

    sensor_size = images.shape[1:3]
    with open(os.path.join(output_pth, 'metadata.json'), 'w') as f:
        json.dump({'sensor_resolution': sensor_size}, f)

    print("✅ Finished", bag_path, "→", output_pth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",default="data/HQF", help="Directory of ROS .bag files")
    parser.add_argument("--event_topic", default="/dvs/events", help="Event topic name")
    parser.add_argument("--image_topic", default="/dvs/image_raw", help="Image topic name")
    args = parser.parse_args()

    for bag_path in glob.glob(os.path.join(args.path, "*.bag")):
        out = os.path.splitext(bag_path)[0]
        print(f"\n📦 Processing {bag_path}")
        try:
            bag_to_npy(bag_path, out, args.event_topic, args.image_topic)
        except Exception as e:
            print("❌ Failed:", bag_path, e)
