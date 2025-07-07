import h5py
import os
import cv2
import numpy as np

def save_images_and_video_from_hdf5(
    hdf5_path,
    dataset_path="/davis/right/image_raw",
    output_dir="output_frames",
    video_path="output_video.mp4",
    fps=30
):
    # 读取数据
    with h5py.File(hdf5_path, "r") as f:
        images = f[dataset_path][()]  # 形状为 (N, H, W)
    
    os.makedirs(output_dir, exist_ok=True)
    num_frames, height, width = images.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=False)

    print(f"📦 Total frames: {num_frames}")
    for i in range(num_frames):
        img = images[i]
        filename = os.path.join(output_dir, f"frame_{i:05d}.png")
        cv2.imwrite(filename, img)
        video_writer.write(img)  # 直接写灰度图即可

    video_writer.release()
    print(f"✅ All frames saved to: {output_dir}")
    print(f"🎞️ Video saved to: {video_path}")

if __name__ == "__main__":
    hdf5_path = "E:/2025/event_flick_flare/EVREAL-main/EVREAL-main/data/MVSEC/outdoor_night1_data.hdf5"  # <- 替换为你的 HDF5 路径
    save_images_and_video_from_hdf5(hdf5_path)
