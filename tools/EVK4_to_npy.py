import os, json, glob, argparse
import time
import numpy as np
import h5py

def hdf5_to_npy(hdf5_path, output_pth, max_duration_sec=None, dt=0.1):
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

    # 读取事件
    with h5py.File(hdf5_path, 'r') as f:
        grp = f.get('CD', f)  # EVK4 HDF5 存储在 group 'CD'
        ev = grp['events']
        x = ev['x'][:].astype(np.int32)
        y = ev['y'][:].astype(np.int32)
        p = ev['p'][:].astype(np.int8)
        ts = ev['t'][:].astype(np.float64) / 1e6  # 转秒
    # print(ts[0])
    # 对齐时间，截断长度
    min_ts = ts.min() if ts.size else 0.0
    ts = ts - min_ts
    if max_duration_sec:
        mask = ts < max_duration_sec
        ts, x, y, p = ts[mask], x[mask], y[mask], p[mask]

    events_ts = ts
    events_xy = np.stack((x, y), axis=-1)
    events_p = p

    # 生成黑色图像帧
    duration = events_ts[-1] if events_ts.size else 0.0
    n_frames = int(np.ceil((duration if max_duration_sec is None else min(duration, max_duration_sec)) / dt))
    h = int(grp.attrs.get('height', 720))
    w = int(grp.attrs.get('width', 1280))

    images = np.zeros((n_frames, h, w, 3), dtype=np.uint8)
    images_ts = np.arange(n_frames) * dt
    images_ts = images_ts.reshape(-1,1)

    if events_ts.size > 0:
        image_event_indices = np.searchsorted(events_ts, images_ts.reshape(-1), side='right').reshape(-1, 1)
        image_event_indices = np.clip(image_event_indices, 0, len(events_ts) - 1)
    else:
        image_event_indices = np.zeros((n_frames, 1), dtype=np.int64)

    # 保存 npy
    np.save(paths['events_ts'], events_ts)
    np.save(paths['events_xy'], events_xy)
    np.save(paths['events_p'], events_p)
    np.save(paths['images'], images)
    np.save(paths['images_ts'], images_ts)
    np.save(paths['image_event_indices'], image_event_indices)

    # 写 metadata
    sensor_size = (h, w)
    with open(metadata_path, 'w') as fd:
        json.dump({"sensor_resolution": list(sensor_size)}, fd)

    print(f"✅ Finished {hdf5_path} → {output_pth}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="HDF5 file or folder")
    parser.add_argument("--max_duration", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.path, '*.h5')) if os.path.isdir(args.path) else [args.path]
    for fp in files:
        out = os.path.splitext(fp)[0]
        print("▶ Processing:", fp)
        start=time.time()
        try:
            hdf5_to_npy(fp, out, max_duration_sec=args.max_duration, dt=args.dt)
        except Exception as e:
            print("❌ Failed:", e)
        print("⏱ Time: %.2fs" % (time.time()-start))
