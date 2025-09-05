import h5py

def explore_h5_structure(name, item):
    """递归遍历 HDF5 文件结构"""
    if isinstance(item, h5py.Group):
        print(f"Group: {name} (contains nested items)")
    elif isinstance(item, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {item.shape}")

def check_h5_structure(file_path):
    try:
        with h5py.File(file_path, 'r') as h5_file:
            h5_file.visititems(explore_h5_structure)
    except Exception as e:
        print(f"Error: {e}")

def read_eventtrueE_df(file_path):
    try:
        with h5py.File(file_path, 'r') as h5_file:
            data = h5_file['eventtrueE/df'][:]
            print("Keys in the HDF5 file:", list(h5_file.keys()))
            print("eventtrueE/df data:", data)
            print("Data shape:", data.shape)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    file_path = "/storage/data/train_data/nova_data/NOVA_MP5/FD-FluxSwap-FHC/preprocessed_regcvn.8_of_1300.h5"
    #check_h5_structure(file_path)
    read_eventtrueE_df(file_path)