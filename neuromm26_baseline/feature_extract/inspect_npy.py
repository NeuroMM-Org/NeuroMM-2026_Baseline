import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Inspect an extracted .npy feature file")
    parser.add_argument("--file-path", "-f", type=str, required=True, help="Absolute or relative path to the .npy file")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"[Error] File not found at: {args.file_path}")
        return

    try:
        # Load the numpy array
        data = np.load(args.file_path)
        
        # Display meta information
        print("=" * 40)
        print(f"File Name : {os.path.basename(args.file_path)}")
        print(f"File Size : {os.path.getsize(args.file_path) / 1024:.2f} KB")
        print("=" * 40)
        print(f"Dimensions: {data.shape}")
        print(f"Data Type : {data.dtype}")
        
        # Display statistical information
        print("-" * 40)
        print(f"Min Value : {data.min():.6f}")
        print(f"Max Value : {data.max():.6f}")
        print(f"Mean Value: {data.mean():.6f}")
        print("-" * 40)
        
        # Display a truncated preview of the content
        # NumPy automatically truncates large arrays with "..." when printing
        print("Data Preview:")
        print(data)
        print("=" * 40)
        
    except Exception as e:
        print(f"[Error] Failed to load the file: {e}")

if __name__ == "__main__":
    main()