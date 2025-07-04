import os
import json
import glob
import argparse
import time

import numpy as np
import cv2
import aedat


def aedat4_to_npy(aedat4_path, output_pth, max_duration_sec=3.0):
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

    xs, ys, ts, ps = [], [], [], []
    image_list, image_ts_list = [], []

    decoder = aedat.Decoder(aedat4_path)
    start_ts = None
    max_t = None

    for packet in decoder:
        if 'events' in packet:
            ev = packet['events']
            if start_ts is None:
                start_ts = ev['t'][0]
                max_t = start_ts + int(max_duration_sec * 1e6)
            mask = ev['t'] <= max_t
            xs.append(ev['x'][mask])
            ys.append(ev['y'][mask])
            ts.append(ev['t'][mask])
            ps.append(ev['p'][mask])

        elif 'frame' in packet:
            # BUG FIX: Check for the existence of the 'data' key before accessing it.
            if 'data' in packet['frame']:
                t = packet['frame']['t']
                if start_ts is None:
                    start_ts = t
                    max_t = start_ts + int(max_duration_sec * 1e6)
                if t <= max_t:
                    img = packet['frame']['data']
                    if img is None:
                        continue
                    if img.ndim == 3:
                        img = img[..., 0]
                    image_list.append(img)
                    image_ts_list.append(t)

    # Flatten arrays
    if ts:
        ts = np.concatenate(ts).astype(np.float64)
        xs = np.concatenate(xs).astype(np.int16)
        ys = np.concatenate(ys).astype(np.int16)
        ps = np.concatenate(ps).astype(np.int8)
    else:
        ts = np.zeros((0,), dtype=np.float64)
        xs = np.zeros((0,), dtype=np.int16)
        ys = np.zeros((0,), dtype=np.int16)
        ps = np.zeros((0,), dtype=np.int8)

    events_ts = ts / 1e6
    events_xy = np.stack((xs, ys), axis=-1).astype(np.int32)
    events_p = ps

    if image_list:
        images = np.stack(image_list, axis=0)
        images_ts = np.array(image_ts_list, dtype=np.float64).reshape(-1, 1) / 1e6
    else:
        images = np.zeros((0, 0, 0), dtype=np.uint8)
        images_ts = np.zeros((0, 1), dtype=np.float64)

    # Align timestamps
    min_ts = 0.0
    if events_ts.size > 0 and images_ts.size > 0:
        min_ts = min(events_ts.min(), images_ts.min())
    elif events_ts.size > 0:
        min_ts = events_ts.min()
    elif images_ts.size > 0:
        min_ts = images_ts.min()


    events_ts -= min_ts
    images_ts -= min_ts

    # Image-event association
    image_event_indices = np.searchsorted(events_ts, images_ts.reshape(-1), side='right')
    image_event_indices = np.clip(image_event_indices, 0, max(len(events_ts) - 1, 0))


    # Save
    np.save(paths['events_ts'], events_ts)
    np.save(paths['events_xy'], events_xy)
    np.save(paths['events_p'], events_p)
    np.save(paths['images'], images)
    np.save(paths['images_ts'], images_ts)
    np.save(paths['image_event_indices'], image_event_indices)

    # Handle case where there might be no images
    if images.size > 0:
        sensor_size = images.shape[1:3]
    elif xs.size > 0:
        # Infer sensor size from events if no frames are present
        sensor_size = (int(ys.max() + 1), int(xs.max() + 1))
    else:
        sensor_size = (0, 0)

    with open(metadata_path, 'w') as f:
        json.dump({"sensor_resolution": sensor_size}, f)

    print(f"✅ Finished {aedat4_path} → {output_pth}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to AEDAT4 file or folder")
    parser.add_argument("--max_duration", type=float, default=3.0)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.path, "*.aedat4")) if os.path.isdir(args.path) else [args.path]

    for fp in files:
        outdir = os.path.splitext(fp)[0]
        os.makedirs(outdir, exist_ok=True)
        print(f"▶ Processing: {fp}")
        start = time.time()
        try:
            aedat4_to_npy(fp, outdir, max_duration_sec=args.max_duration)
        except Exception as e:
            print(f"❌ Failed {fp}: {e}")
        print(f"⏱️ Time elapsed: {time.time() - start:.2f} seconds")
