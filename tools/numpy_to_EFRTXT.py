import os
import argparse
import numpy as np
import json
import shutil


def npy_to_txt_corrected(npy_dir):
    # === Load data ===
    events_ts = np.load(os.path.join(npy_dir, "events_ts.npy"))  # (N,)
    events_xy = np.load(os.path.join(npy_dir, "events_xy.npy"))  # (N, 2)
    events_p = np.load(os.path.join(npy_dir, "events_p.npy"))    # (N,)

    with open(os.path.join(npy_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
        height, width = metadata["sensor_resolution"]  # 👈 注意顺序是 [H, W]

    # === 修正分辨率顺序（宽 高） ===
    lines = [f"{width} {height}"]

    # === 将时间戳转换为整数毫秒 ===
    t_ms = (events_ts * 1000000).astype(np.int64)  # 👈 转为整数毫秒

    # === 生成事件行 ===
    lines += [f"{t} {x} {y} {p}" for t, (x, y), p in zip(t_ms, events_xy, events_p)]

    # === 写入 events_raw.txt ===
    raw_path = os.path.join(npy_dir, "events_raw.txt")
    with open(raw_path, "w") as f:
        f.write('\n'.join(lines) + '\n')
    print(f"✅ Saved: {raw_path}")

    # === 写入 bias.txt ===
    bias_path = os.path.join(npy_dir, "bias.txt")
    row = '0 ' * width
    bias_lines = [row.strip()] * height  # height 行，每行 width 个 0
    with open(bias_path, "w") as f:
        f.write('\n'.join(bias_lines) + '\n')
    print(f"✅ Saved: {bias_path}")

    # === 复制为 events_raw_no_bias.txt ===
    no_bias_path = os.path.join(npy_dir, "events_raw_no_bias.txt")
    shutil.copyfile(raw_path, no_bias_path)
    print(f"✅ Saved: {no_bias_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to folder containing .npy files and metadata.json")
    args = parser.parse_args()

    npy_to_txt_corrected(args.path)
