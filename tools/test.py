import os
import numpy as np
import cv2
import argparse
import json

def load_data(folder):
    events_ts = np.load(os.path.join(folder, "events_ts.npy"))
    events_xy = np.load(os.path.join(folder, "events_xy.npy"))
    events_p = np.load(os.path.join(folder, "events_p.npy"))

    images = np.load(os.path.join(folder, "images.npy"))
    images_ts = np.load(os.path.join(folder, "images_ts.npy"))

    with open(os.path.join(folder, "metadata.json")) as f:
        meta = json.load(f)
    sensor_size = meta["sensor_resolution"]

    return events_ts, events_xy, events_p, images, images_ts, sensor_size

def show_frame(image, title="Frame"):
    image_to_show = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image_to_show = image_to_show.astype(np.uint8)
    cv2.imshow(title, image_to_show)
    cv2.waitKey(0)

def show_events(events_xy, events_p, canvas_shape, title="Events", point_size=1):
    canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=np.uint8)

    for (x, y), p in zip(events_xy, events_p):
        if 0 <= y < canvas_shape[0] and 0 <= x < canvas_shape[1]:
            color = (0, 0, 255) if p > 0 else (255, 0, 0)  # Red for pos, Blue for neg
            cv2.circle(canvas, (x, y), point_size, color, -1)

    cv2.imshow(title, canvas)
    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",default="E:/2025/event_flick_flare/datasets/DAVIS/test/full_sixFlare-2025_06_25_14_50_08", help="Path to folder containing .npy files")
    parser.add_argument("--event_window", type=float, default=0.05, help="Duration (sec) of events to show")
    args = parser.parse_args()

    print(f"Loading from {args.folder}")
    events_ts, events_xy, events_p, images, images_ts, sensor_size = load_data(args.folder)

    print("Showing first image frame...")
    if images.shape[0] > 0:
        show_frame(images[0, :, :], title="First Frame")

    print(f"Showing events in first {args.event_window} seconds...")
    mask = events_ts <= args.event_window
    show_events(events_xy[mask], events_p[mask], sensor_size, title=f"Events in {args.event_window}s")
