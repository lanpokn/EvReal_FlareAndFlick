import cv2
import os
import math
import argparse
import numpy as np

def resize_frame(frame, size):
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

def load_videos(video_paths):
    caps = [cv2.VideoCapture(p) for p in video_paths]
    fps_list = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
    widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    return caps, fps_list, widths, heights, frame_counts

def auto_grid(num_videos):
    grid_h = math.floor(math.sqrt(num_videos))
    grid_w = math.ceil(num_videos / grid_h)
    return grid_h, grid_w

def compose_grid(frames, grid_h, grid_w, frame_size, fill_value=0):
    # Fill up to full grid with black frames if needed
    blank = np.full((frame_size[1], frame_size[0], 3), fill_value, dtype=np.uint8)
    frames += [blank] * (grid_h * grid_w - len(frames))

    rows = []
    for i in range(grid_h):
        row = frames[i*grid_w:(i+1)*grid_w]
        rows.append(np.hstack(row))
    return np.vstack(rows)

def combine_videos(video_paths, output_path, resize_to=(320, 240), max_frames=None):
    caps, fps_list, widths, heights, frame_counts = load_videos(video_paths)
    fps = int(min(fps_list)) if all(fps_list) else 24
    total_frames = min([cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps])
    num_videos = len(caps)
    max_frames = min(max_frames or int(total_frames), int(total_frames))

    grid_h, grid_w = auto_grid(num_videos)

    out_w = resize_to[0] * grid_w
    out_h = resize_to[1] * grid_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    print(f"▶ Combining {num_videos} videos into {grid_h}x{grid_w} grid. Total frames: {max_frames}")
    print(f"▶ Output size: {out_w}x{out_h}, FPS: {fps}")

    for frame_idx in range(max_frames):
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((resize_to[1], resize_to[0], 3), dtype=np.uint8)
            else:
                if frame.ndim == 2 or frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = resize_frame(frame, resize_to)
            frames.append(frame)
        combined = compose_grid(frames, grid_h, grid_w, resize_to)
        out.write(combined)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{max_frames}")

    for cap in caps:
        cap.release()
    out.release()
    print(f"✅ Done! Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", help="Paths to video files")
    parser.add_argument("--output", default="combined_output.mp4", help="Output video file name")
    parser.add_argument("--width", type=int, default=320, help="Resize width of each video")
    parser.add_argument("--height", type=int, default=240, help="Resize height of each video")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process (default: full length)")
    args = parser.parse_args()

    combine_videos(args.videos, args.output, resize_to=(args.width, args.height), max_frames=args.max_frames)
