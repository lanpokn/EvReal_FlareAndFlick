import os
import json
import glob
import argparse
import time

import numpy as np
import cv2
import aedat


def aedat4_to_npy(aedat4_path, output_pth, max_duration_sec=None):
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

    # 1. Read ALL data from the file first.
    decoder = aedat.Decoder(aedat4_path)
    # print(decoder)
    for packet in decoder:
        # print(packet)
        if 'events' in packet:
            ev = packet['events']
            xs.append(ev['x'])
            ys.append(ev['y'])
            ts.append(ev['t'])
            ps.append(ev['p'])

        # THE FIX: Check for 'frames' (plural), not 'frame' (singular).
        # This was the root cause of images not being read.
        elif 'frame' in packet:
            # Also use 'frames' (plural) to access the data within the packet.
            frame_packet = packet['frame']
            # The 'data' key might not exist in every frame packet (e.g. metadata only)
            if 'pixels' in frame_packet:
                img = frame_packet['pixels']
                if img is None:
                    continue
                # if img.ndim == 3:
                #     img = img[..., 0]
                image_list.append(img)
                image_ts_list.append(frame_packet['t'])

    # --- Data preparation, alignment, and filtering ---
    if not ts and not image_list:
        print(f"❌ No events or frames found in {aedat4_path}. Aborting.")
        return

    # Flatten event arrays
    if ts:
        ts = np.concatenate(ts).astype(np.float64)
        xs = np.concatenate(xs).astype(np.int16)
        ys = np.concatenate(ys).astype(np.int16)
        ps = np.concatenate(ps).astype(np.int8)
        events_ts = ts / 1e6
        events_xy = np.stack((xs, ys), axis=-1).astype(np.int32)
        events_p = ps
    else:
        events_ts = np.zeros((0,), dtype=np.float64)
        events_xy = np.zeros((0, 2), dtype=np.int32)
        events_p = np.zeros((0,), dtype=np.int8)

    # Stack image arrays
    if image_list:
        images = np.stack(image_list, axis=0)
        images_ts = np.array(image_ts_list, dtype=np.float64).reshape(-1, 1) / 1e6
    else:
        images = np.zeros((0, 0, 0), dtype=np.uint8)
        images_ts = np.zeros((0, 1), dtype=np.float64)

    # 2. Find the true start time from ALL collected data.
    min_ts = 0.0
    has_events = events_ts.size > 0
    has_images = images_ts.size > 0

    if has_events and has_images:
        min_ts = min(events_ts.min(), images_ts.min())
    elif has_events:
        min_ts = events_ts.min()
    elif has_images:
        min_ts = images_ts.min()

    # 3. Align all timestamps to start from t=0.
    events_ts -= min_ts
    images_ts -= min_ts

    # 4. Apply the duration cutoff on the aligned timestamps.
    if max_duration_sec is not None:
        if has_events:
            event_mask = events_ts < max_duration_sec
            events_ts = events_ts[event_mask]
            events_xy = events_xy[event_mask]
            events_p = events_p[event_mask]
        
        if has_images:
            image_mask = images_ts.flatten() < max_duration_sec
            images = images[image_mask]
            images_ts = images_ts[image_mask]

    # Image-event association (must be done AFTER filtering)
    if images.size > 0 and events_ts.size > 0:
        image_event_indices = np.searchsorted(events_ts, images_ts.reshape(-1), side='right').reshape(-1, 1)
        # Clip is essential for cases where the last frame is past the last event
        image_event_indices = np.clip(image_event_indices, 0, len(events_ts) - 1)
    else:
        image_event_indices = np.zeros(len(images), dtype=np.int64)
    # print(image_event_indices)
    # --- Save ---
    np.save(paths['events_ts'], events_ts)
    np.save(paths['events_xy'], events_xy)
    np.save(paths['events_p'], events_p)
    np.save(paths['images'], images)
    np.save(paths['images_ts'], images_ts)
    np.save(paths['image_event_indices'], image_event_indices)

    # Determine sensor size for metadata
    sensor_size = (0, 0)
    if images.size > 0:
        sensor_size = images.shape[1:3]
    elif events_xy.size > 0:
        # Infer from events if no frames exist
        sensor_size = (int(events_xy[:, 1].max() + 1), int(events_xy[:, 0].max() + 1))
        
    with open(metadata_path, 'w') as f:
        json.dump({"sensor_resolution": list(sensor_size)}, f)

    print(f"✅ Finished {aedat4_path} → {output_pth} (Frames: {len(images)}, Events: {len(events_ts)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path",default="E:/2025/event_flick_flare/datasets/DAVIS/test/full_sixFlare-2025_06_25_14_50_08.aedat4", help="Path to AEDAT4 file or folder")
    # Set a more reasonable default. If you want the full file, just omit the argument.
    parser.add_argument("--max_duration", type=float, default=None, 
                        help="Maximum duration to save in seconds. If not set, saves the entire file.")
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
            import traceback
            traceback.print_exc() # This will print more detailed error info
        print(f"⏱️ Time elapsed: {time.time() - start:.2f} seconds")