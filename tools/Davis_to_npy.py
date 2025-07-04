# import os
# import json
# import glob
# import argparse

# import numpy as np
# from dv import AedatFile
# import cv2


# def aedat4_to_npy(aedat4_path, output_pth, max_duration_sec=3.0):
#     metadata_path = os.path.join(output_pth, 'metadata.json')
#     events_ts_path = os.path.join(output_pth, 'events_ts.npy')
#     events_xy_path = os.path.join(output_pth, 'events_xy.npy')
#     events_p_path = os.path.join(output_pth, 'events_p.npy')
#     images_path = os.path.join(output_pth, 'images.npy')
#     images_ts_path = os.path.join(output_pth, 'images_ts.npy')
#     image_event_indices_path = os.path.join(output_pth, 'image_event_indices.npy')

#     xs, ys, ts, ps = [], [], [], []
#     image_list, image_ts_list = [], []

#     with AedatFile(aedat4_path) as f:
#         # --- Read events ---
#         if 'events' in f.names:
#             print(f"Reading events from {aedat4_path}...")
#             event_stream = f['events']
#             count = 0
#             start_ts = None
#             for event in event_stream:
#                 if start_ts is None:
#                     start_ts = event.timestamp
#                 current_ts = event.timestamp
#                 if (current_ts - start_ts) / 1e6 > max_duration_sec:  # timestamps in microseconds
#                     break
#                 xs.append(event.x)
#                 ys.append(event.y)
#                 ps.append(event.polarity)
#                 ts.append(current_ts)
#                 count += 1
#             print(f"Total events read: {count}")

#         # --- Read frames ---
#         if 'frames' in f.names:
#             print(f"Reading frames from {aedat4_path}...")
#             frame_stream = f['frames']
#             count = 0
#             start_frame_ts = None
#             for frame in frame_stream:
#                 if start_frame_ts is None:
#                     start_frame_ts = frame.timestamp
#                 current_ts = frame.timestamp
#                 if (current_ts - start_frame_ts) / 1e6 > max_duration_sec:
#                     break
#                 image_ts_list.append(current_ts)
#                 img = frame.image
#                 if len(img.shape) == 3 and img.shape[2] == 3:
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 image_list.append(img)
#                 count += 1
#             print(f"Total frames read: {count}")

#     # --- Events ---
#     events_ts = np.array(ts, dtype=np.float64) / 1e6  # convert to seconds
#     events_xy = np.array([xs, ys], dtype=np.int32).T
#     events_p = np.array(ps, dtype=np.int8)

#     # --- Frames ---
#     if len(image_list) == 0:
#         print("Warning: no frames found, writing empty image arrays.")
#         images = np.zeros((0, 0, 0), dtype=np.uint8)
#         images_ts = np.zeros((0, 1), dtype=np.float64)
#     else:
#         images = np.stack(image_list, axis=0)
#         images_ts = np.array(image_ts_list, dtype=np.float64).reshape(-1, 1) / 1e6  # seconds

#     # --- Align timestamps ---
#     if events_ts.size > 0 and images_ts.size > 0:
#         min_ts = min(events_ts.min(), images_ts.min())
#     elif events_ts.size > 0:
#         min_ts = events_ts.min()
#     elif images_ts.size > 0:
#         min_ts = images_ts.min()
#     else:
#         min_ts = 0

#     events_ts -= min_ts
#     images_ts -= min_ts

#     # --- Image-event association ---
#     image_event_indices = np.searchsorted(events_ts, images_ts.reshape(-1), side='right') - 1
#     image_event_indices = np.clip(image_event_indices, 0, max(len(events_ts) - 1, 0))

#     # --- Save ---
#     np.save(events_ts_path, events_ts, allow_pickle=False, fix_imports=False)
#     np.save(events_xy_path, events_xy, allow_pickle=False, fix_imports=False)
#     np.save(events_p_path, events_p, allow_pickle=False, fix_imports=False)
#     np.save(images_path, images, allow_pickle=False, fix_imports=False)
#     np.save(images_ts_path, images_ts, allow_pickle=False, fix_imports=False)
#     np.save(image_event_indices_path, image_event_indices, allow_pickle=False, fix_imports=False)

#     sensor_size = images.shape[1:3] if images.shape[0] > 0 else (0, 0)
#     with open(metadata_path, 'w') as f:
#         json.dump({"sensor_resolution": sensor_size}, f)

#     print(f"✅ Finished {aedat4_path} → {output_pth}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("path", help="Path to AEDAT4 file or folder")
#     parser.add_argument("--max_duration", type=float, default=3.0, help="Max duration to read (in seconds)")
#     args = parser.parse_args()

#     if os.path.isdir(args.path):
#         files = glob.glob(os.path.join(args.path, "*.aedat4"))
#     else:
#         files = [args.path]

#     for file_path in files:
#         output_dir = os.path.splitext(file_path)[0]
#         os.makedirs(output_dir, exist_ok=True)
#         try:
#             aedat4_to_npy(file_path, output_dir, max_duration_sec=args.max_duration)
#         except Exception as e:
#             print(f"❌ Failed processing {file_path}")
#             print(e)
import os
import json
import glob
import argparse

import numpy as np
from dv import AedatFile
import cv2


def aedat4_to_npy(aedat4_path, output_pth, max_duration_sec=3.0):
    metadata_path = os.path.join(output_pth, 'metadata.json')
    events_ts_path = os.path.join(output_pth, 'events_ts.npy')
    events_xy_path = os.path.join(output_pth, 'events_xy.npy')
    events_p_path = os.path.join(output_pth, 'events_p.npy')
    images_path = os.path.join(output_pth, 'images.npy')
    images_ts_path = os.path.join(output_pth, 'images_ts.npy')
    image_event_indices_path = os.path.join(output_pth, 'image_event_indices.npy')

    image_list, image_ts_list = [], []

    with AedatFile(aedat4_path) as f:
        # --- Read events ---
        if 'events' in f.names:
            print(f"Reading events from {aedat4_path}...")
            event_stream = f['events']
            xs, ys, ts, ps = [], [], [], []
            count = 0
            start_ts = None
            for event in event_stream:
                if start_ts is None:
                    start_ts = event.timestamp
                current_ts = event.timestamp
                if (current_ts - start_ts) / 1e6 > max_duration_sec:
                    break
                xs.append(event.x)
                ys.append(event.y)
                ts.append(event.timestamp)
                ps.append(event.polarity)
                count += 1

            xs = np.array(xs, dtype=np.int16)
            ys = np.array(ys, dtype=np.int16)
            ts = np.array(ts, dtype=np.float64)
            ps = np.array(ps, dtype=np.int8)
            print(f"Total events read: {len(xs)}")
        # --- Read frames ---
        if 'frames' in f.names:
            print(f"Reading frames from {aedat4_path}...")
            frame_stream = f['frames']
            count = 0
            start_frame_ts = None
            for frame in frame_stream:
                if start_frame_ts is None:
                    start_frame_ts = frame.timestamp
                if (frame.timestamp - start_frame_ts) / 1e6 > max_duration_sec:
                    break
                image_ts_list.append(frame.timestamp)
                img = frame.image
                if img.ndim == 3:
                    img = img[..., 0]  # assume grayscale
                image_list.append(img)
                count += 1
            print(f"Total frames read: {count}")

    # --- Events ---
    if len(ts) > 0:
        events_ts = ts / 1e6  # seconds
        events_xy = np.stack((xs, ys), axis=-1)
        events_p = ps
    else:
        events_ts = np.zeros((0,), dtype=np.float64)
        events_xy = np.zeros((0, 2), dtype=np.int32)
        events_p = np.zeros((0,), dtype=np.int8)

    # --- Frames ---
    if len(image_list) == 0:
        print("Warning: no frames found, writing empty image arrays.")
        images = np.zeros((0, 0, 0), dtype=np.uint8)
        images_ts = np.zeros((0, 1), dtype=np.float64)
    else:
        images = np.stack(image_list, axis=0)
        images_ts = np.array(image_ts_list, dtype=np.float64).reshape(-1, 1) / 1e6

    # --- Align timestamps ---
    all_ts = []
    if events_ts.size > 0:
        all_ts.append(events_ts.min())
    if images_ts.size > 0:
        all_ts.append(images_ts.min())
    min_ts = min(all_ts) if all_ts else 0.0

    events_ts -= min_ts
    images_ts -= min_ts

    # --- Image-event association ---
    image_event_indices = np.searchsorted(events_ts, images_ts.reshape(-1), side='right') - 1
    image_event_indices = np.clip(image_event_indices, 0, max(len(events_ts) - 1, 0))

    # --- Save ---
    np.save(events_ts_path, events_ts, allow_pickle=False, fix_imports=False)
    np.save(events_xy_path, events_xy, allow_pickle=False, fix_imports=False)
    np.save(events_p_path, events_p, allow_pickle=False, fix_imports=False)
    np.save(images_path, images, allow_pickle=False, fix_imports=False)
    np.save(images_ts_path, images_ts, allow_pickle=False, fix_imports=False)
    np.save(image_event_indices_path, image_event_indices, allow_pickle=False, fix_imports=False)

    sensor_size = images.shape[1:3] if images.shape[0] > 0 else (0, 0)
    with open(metadata_path, 'w') as f:
        json.dump({"sensor_resolution": sensor_size}, f)

    print(f"✅ Finished {aedat4_path} → {output_pth}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to AEDAT4 file or folder")
    parser.add_argument("--max_duration", type=float, default=3.0, help="Max duration to read (in seconds)")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, "*.aedat4"))
    else:
        files = [args.path]

    for file_path in files:
        output_dir = os.path.splitext(file_path)[0]
        os.makedirs(output_dir, exist_ok=True)
        try:
            aedat4_to_npy(file_path, output_dir, max_duration_sec=args.max_duration)
        except Exception as e:
            print(f"❌ Failed processing {file_path}")
            print(e)
