import os
import argparse
import numpy as np


def txt_to_npy(txt_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(txt_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(txt_path, "r") as f:
        lines = f.readlines()

    # 忽略第一行分辨率
    data_lines = lines[1:]

    # 解析数据
    ts_list = []
    xy_list = []
    p_list = []

    for line in data_lines:
        if not line.strip():
            continue
        try:
            t_str, x_str, y_str, p_str = line.strip().split()
            t = int(t_str)
            x = int(x_str)
            y = int(y_str)
            p = int(p_str)
        except ValueError:
            print(f"⚠️ Skipping malformed line: {line}")
            continue

        ts_list.append(t / 1000.0)  # 转为秒
        xy_list.append((x, y))
        p_list.append(p)

    # 转为 numpy 并保存
    ts = np.array(ts_list, dtype=np.float64)
    xy = np.array(xy_list, dtype=np.int32)
    p = np.array(p_list, dtype=np.int8)

    np.save(os.path.join(output_dir, "events_ts.npy"), ts)
    np.save(os.path.join(output_dir, "events_xy.npy"), xy)
    np.save(os.path.join(output_dir, "events_p.npy"), p)

    print(f"✅ Converted {txt_path} → events_ts.npy, events_xy.npy, events_p.npy in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("txt_path", help="Path to events_raw.txt or events_raw_no_bias.txt")
    parser.add_argument("--output_dir", default=None, help="Directory to save npy files (default: same as input)")
    args = parser.parse_args()

    txt_to_npy(args.txt_path, args.output_dir)
