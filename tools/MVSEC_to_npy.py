import os
import json
import argparse
import time
import numpy as np
import h5py
import cv2


def mvsec_hdf5_to_npy(hdf5_path, output_pth, start_frame=0, end_frame=None):
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

    with h5py.File(hdf5_path, 'r') as f:
        # Load image data (left camera)
        images = f['davis/left/image_raw'][start_frame:end_frame]
        images_ts = f['davis/left/image_raw_ts'][start_frame:end_frame]
        image_event_inds = f['davis/left/image_raw_event_inds'][start_frame:end_frame]

        # Load event data (left camera)
        events = f['davis/left/events']
        if end_frame is None:
            end_idx = len(images_ts) - 1
        else:
            end_idx = end_frame - start_frame - 1
        start_ev_idx = image_event_inds[0]
        end_ev_idx = image_event_inds[end_idx] if end_idx < len(image_event_inds) else events.shape[0]

        events_slice = events[start_ev_idx:end_ev_idx]
        xs = events_slice[:, 0].astype(np.int16)
        ys = events_slice[:, 1].astype(np.int16)
        ts = events_slice[:, 2].astype(np.float64) / 1e6  # seconds
        ps = events_slice[:, 3].astype(np.int8)

        # Align time
        min_ts = min(ts.min(), images_ts[0] / 1e6)
        ts -= min_ts
        images_ts = images_ts / 1e6 - min_ts

        # Event array formatting
        events_ts = ts
        events_xy = np.stack((xs, ys), axis=-1)
        events_p = ps

        # Convert grayscale images to 3-channel BGR
        images_rgb = np.stack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in images], axis=0)

        # Event indices per image
        image_event_indices = np.searchsorted(events_ts, images_ts.reshape(-1), side='right')
        image_event_indices = np.clip(image_event_indices, 0, len(events_ts) - 1)

        # Save .npy
        np.save(paths['events_ts'], events_ts)
        np.save(paths['events_xy'], events_xy)
        np.save(paths['events_p'], events_p)
        np.save(paths['images'], images_rgb)
        np.save(paths['images_ts'], images_ts.reshape(-1, 1))
        np.save(paths['image_event_indices'], image_event_indices.reshape(-1, 1))

        # Metadata
        sensor_size = list(images.shape[1:3])
        with open(metadata_path, 'w') as fmeta:
            json.dump({"sensor_resolution": sensor_size}, fmeta)

    print(f"✅ Converted {hdf5_path} → {output_pth}")
    print(f"Images: {len(images)}, Events: {len(events_ts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to MVSEC HDF5 file")
    parser.add_argument("--output", default=None, help="Output directory (default: same as input name)")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index (exclusive)")
    args = parser.parse_args()

    out_dir = args.output or os.path.splitext(args.path)[0]
    t0 = time.time()
    mvsec_hdf5_to_npy(args.path, out_dir, start_frame=args.start_frame, end_frame=args.end_frame)
    print(f"⏱️ Time elapsed: {time.time() - t0:.2f} seconds")