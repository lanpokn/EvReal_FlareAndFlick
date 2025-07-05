import os
import cv2
import argparse
import numpy as np
from glob import glob
import json

def save_video(frames, output_path, fps=30):
    if len(frames) == 0:
        print("⚠️ No frames to write.")
        return
    h, w = frames[0].shape[:2]
    is_color = len(frames[0].shape) == 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=is_color)
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()
    print(f"✅ Video saved to {output_path}")


def read_images_from_folder(folder):
    images = []
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, ext)))
    files = sorted(files)
    for file in files:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
    return images


def read_images_from_npy(npy_path):
    # channel_orders = {
    #     "BGR": (0, 1, 2),
    #     "RGB": (2, 1, 0),
    #     "GBR": (1, 2, 0),
    #     "RBG": (2, 0, 1),
    #     "GRB": (1, 0, 2),
    #     "BRG": (0, 2, 1)
    # }
    change = (2, 1, 0)
    images = np.load(npy_path)
    frames = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img[:, :, change]

        frames.append(img)
    return frames


def visualize_events_to_frames(events_xy, events_ts, events_p, resolution, dt=0.1):
    """将事件数据转为一系列红蓝图帧，每 dt 秒一帧"""
    assert len(events_xy) == len(events_ts) == len(events_p)
    events_ts = np.array(events_ts)
    duration = events_ts[-1]
    num_frames = int(np.ceil(duration / dt))
    frames = []

    h, w = resolution
    for i in range(num_frames):
        t_start = i * dt
        t_end = (i + 1) * dt
        mask = (events_ts >= t_start) & (events_ts < t_end)
        if not np.any(mask):
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            continue
        xy = events_xy[mask]
        p = events_p[mask]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for (x, y), polarity in zip(xy, p):
            if 0 <= x < w and 0 <= y < h:
                if polarity > 0:
                    frame[y, x, 2] = 255  # red
                else:
                    frame[y, x, 0] = 255  # blue
        frames.append(frame)
    return frames


def process_input(input_path, output_path, fps=30, event_dt=0.1):
    if os.path.isdir(input_path):
        print("📁 Input is folder of images.")
        frames = read_images_from_folder(input_path)

    elif input_path.endswith("images.npy"):
        print("📦 Input is image numpy array.")
        frames = read_images_from_npy(input_path)

    elif input_path.endswith("events_xy.npy"):
        print("⚡ Input is event data.")
        dir_path = os.path.dirname(input_path)
        xy = np.load(os.path.join(dir_path, "events_xy.npy"))
        ts = np.load(os.path.join(dir_path, "events_ts.npy"))
        p = np.load(os.path.join(dir_path, "events_p.npy"))
        meta_path = os.path.join(dir_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            resolution = tuple(meta.get("sensor_resolution", (180, 240)))
        else:
            resolution = (180, 240)  # fallback
        frames = visualize_events_to_frames(xy, ts, p, resolution, dt=event_dt)
    else:
        raise ValueError("Unsupported input path type")

    save_video(frames, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input path (folder or .npy)")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")
    parser.add_argument("--dt", type=float, default=0.01, help="Time per event frame (for events only)")
    args = parser.parse_args()

    process_input(args.input, args.output, fps=args.fps, event_dt=args.dt)
