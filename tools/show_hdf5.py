import h5py
import sys

def explore_hdf5_structure(file_path):
    def print_attrs(name, obj):
        print(f"/n🧩 Path: {name}")
        if isinstance(obj, h5py.Dataset):
            print(f"  Type: Dataset")
            print(f"  Shape: {obj.shape}")
            print(f"  Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  Type: Group")
        # 打印属性（如果有）
        for key, val in obj.attrs.items():
            print(f"  Attribute - {key}: {val}")

    with h5py.File(file_path, 'r') as f:
        print(f"📂 Exploring HDF5 File: {file_path}")
        f.visititems(print_attrs)

if __name__ == "__main__":
    # 用法示例：python explore_hdf5.py path/to/your_file.h5
    file_path = "E:/2025/event_flick_flare/EVREAL-main/EVREAL-main/data/MVSEC/outdoor_night1_data.hdf5"  # 可手动修改
    explore_hdf5_structure(file_path)
