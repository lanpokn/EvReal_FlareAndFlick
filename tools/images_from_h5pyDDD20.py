import h5py
import os
import cv2
import numpy as np

def save_davis_left_images_and_video(hdf5_path, output_dir="output_frames/davis_left_output", fps=30):
    # 打开 HDF5 文件
    with h5py.File(hdf5_path, 'r') as f:
        # 读取图像数据
        images = f['davis/left/image_raw'][()]
        print(f"📷 Loaded {images.shape[0]} frames from /davis/left/image_raw")

    os.makedirs(output_dir, exist_ok=True)
    num_frames, height, width = images.shape

    # 设置视频输出路径
    video_path = os.path.join(output_dir, "davis_left_video.mp4")

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=False)

    for i, img in enumerate(images):
        filename = os.path.join(output_dir, f"frame_{i:05d}.png")
        cv2.imwrite(filename, img)      # 保存图像
        video_writer.write(img)         # 写入视频帧

    video_writer.release()
    print(f"✅ Images saved to: {output_dir}")
    print(f"🎞️ Video saved to: {video_path}")

if __name__ == "__main__":
    hdf5_path = "E:/2025/event_flick_flare/EVREAL-main/EVREAL-main/data/DDD20ITSC/rec1498946027.hdf5/rec1498946027.hdf5"
    save_davis_left_images_and_video(hdf5_path)
