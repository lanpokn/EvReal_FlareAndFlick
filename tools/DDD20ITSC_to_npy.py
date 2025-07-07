import os
import json
import argparse
import time
import numpy as np
import h5py
import cv2


def extract_event_dataset(hdf5_path, output_pth, interval=0.05, resolution=(346, 260)):
    os.makedirs(output_pth, exist_ok=True)
    metadata_path = os.path.join(output_pth, 'metadata.json')
    paths = {
        'events_ts': os.path.join(output_pth, 'events_ts.npy'),
        'events_xy': os.path.join(output_pth, 'events_xy.npy'),
        'events_p': os.path.join(output_pth, 'events_p.npy'),
        'images': os.path.join(output_pth, 'images.npy'),
        'images_ts': os.path.join(output_pth, 'images_ts.npy'),
        'image_event_indices': os.path.join(output_pth, 'image_event_indices.npy'),
    }

    print("📂 Loading HDF5...")
    with h5py.File(hdf5_path, 'r') as f:
        dvs_data = f['dvs/data'][:]  # (N,) dtype=object，每个元素是长度3数组或list
        dvs_ts = f['dvs/timestamp'][:]

    # 关键修改：用vstack把list/array堆叠成二维数组
    dvs_data_np = np.vstack(dvs_data).astype(np.int16)  # shape (N, 3)

    xs = dvs_data_np[:, 0]
    ys = dvs_data_np[:, 1]
    ps = dvs_data_np[:, 2].astype(np.int8)
    ts = dvs_ts.astype(np.float64) / 1e6  # 转秒

    # Align time
    min_ts = ts.min()
    ts -= min_ts

    # Simulate image timestamps at fixed interval (e.g., 0.05s)
    max_ts = ts.max()
    images_ts = np.arange(0, max_ts, interval)
    image_event_indices = np.searchsorted(ts, images_ts, side='right')

    # Generate dummy black images
    height, width = resolution[1], resolution[0]
    dummy_img = np.zeros((height, width, 3), dtype=np.uint8)
    images = np.stack([dummy_img.copy() for _ in images_ts], axis=0)

    # Save npy
    np.save(paths['events_ts'], ts)
    np.save(paths['events_xy'], np.stack([xs, ys], axis=-1))
    np.save(paths['events_p'], ps)
    np.save(paths['images'], images)
    np.save(paths['images_ts'], images_ts.reshape(-1, 1))
    np.save(paths['image_event_indices'], image_event_indices.reshape(-1, 1))

    # Metadata
    with open(metadata_path, 'w') as fmeta:
        json.dump({"sensor_resolution": [width, height]}, fmeta)

    print(f"✅ Converted {hdf5_path} → {output_pth}")
    print(f"Images: {len(images)}, Events: {len(ts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to HDF5 file")
    parser.add_argument("--output", default=None, help="Output directory (default: same as input name)")
    parser.add_argument("--interval", type=float, default=0.05, help="Time interval between fake frames (seconds)")
    parser.add_argument("--width", type=int, default=346, help="Sensor width")
    parser.add_argument("--height", type=int, default=260, help="Sensor height")
    args = parser.parse_args()

    out_dir = args.output or os.path.splitext(args.path)[0]
    t0 = time.time()
    extract_event_dataset(args.path, out_dir, interval=args.interval, resolution=(args.width, args.height))
    print(f"⏱️ Time elapsed: {time.time() - t0:.2f} seconds")
