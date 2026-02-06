"""
Convert LIBERO dataset from HDF5 format to Atlas format

LIBERO原始格式: HDF5文件，每个任务一个文件
Atlas格式: episode目录结构，包含images/和actions文件
"""
import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../dataset/LIBERO'))
try:
    from libero.libero import benchmark, get_libero_path
except ImportError:
    print("Warning: Could not import libero. Make sure LIBERO is installed.")
    print("Run: cd dataset/LIBERO && pip install -e .")


def convert_hdf5_to_atlas_format(
    hdf5_path: str,
    output_dir: str,
    benchmark_name: str = "libero_10",
    task_id: int = None,
    split: str = "train"
):
    """
    Convert LIBERO HDF5 dataset to Atlas format
    
    Args:
        hdf5_path: Path to LIBERO HDF5 file
        output_dir: Output directory for Atlas format data
        benchmark_name: Name of the benchmark (e.g., "libero_10")
        task_id: Specific task ID to convert (None = all tasks)
        split: "train" or "val"
    """
    # Create output directory
    output_split_dir = os.path.join(output_dir, split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    # Load HDF5 file
    print(f"Loading HDF5 file: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        # Get dataset structure
        demos = list(f['data'].keys())
        print(f"Found {len(demos)} demonstrations")
        
        # Get task description from benchmark
        task_description = "Unknown task"
        try:
            benchmark_dict = benchmark.get_benchmark_dict()
            if benchmark_name in benchmark_dict:
                task_suite = benchmark_dict[benchmark_name]()
                if task_id is not None and task_id < len(task_suite.tasks):
                    task_description = task_suite.get_task(task_id).language
                    print(f"Task description: {task_description}")
        except Exception as e:
            print(f"Warning: Could not get task description: {e}")
        
        episode_idx = 0
        
        for demo_key in tqdm(demos, desc="Converting demonstrations"):
            demo_group = f['data'][demo_key]
            
            # Create episode directory
            episode_dir = os.path.join(output_split_dir, f"episode_{episode_idx:06d}")
            os.makedirs(episode_dir, exist_ok=True)
            images_dir = os.path.join(episode_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Extract observations (images)
            obs_group = demo_group['obs']
            
            # Get workspace camera images
            workspace_key = None
            wrist_key = None
            
            # Try different possible keys for camera images
            for key in obs_group.keys():
                if 'workspace' in key.lower() or 'agentview' in key.lower() or 'rgb' in key.lower():
                    if workspace_key is None:
                        workspace_key = key
                if 'wrist' in key.lower() or 'eye_in_hand' in key.lower():
                    wrist_key = key
            
            # If no workspace key found, try to find any image-like data
            if workspace_key is None:
                for key in obs_group.keys():
                    if len(obs_group[key].shape) >= 3:  # Likely an image
                        workspace_key = key
                        break
            
            if workspace_key is None:
                print(f"Warning: No workspace camera found in demo {demo_key}, skipping")
                continue
            
            workspace_images = obs_group[workspace_key]
            num_frames = workspace_images.shape[0]
            
            # Extract wrist images if available
            wrist_images = None
            if wrist_key is not None:
                wrist_images = obs_group[wrist_key]
            
            # Extract actions
            actions = demo_group['actions'][:]  # Shape: (num_frames, action_dim)
            
            # Ensure actions are 7-DOF (pad or truncate if needed)
            if actions.shape[1] < 7:
                # Pad with zeros
                padding = np.zeros((actions.shape[0], 7 - actions.shape[1]))
                actions = np.concatenate([actions, padding], axis=1)
            elif actions.shape[1] > 7:
                # Truncate to 7
                actions = actions[:, :7]
            
            # Save images and actions
            for frame_idx in range(num_frames):
                # Save workspace image
                workspace_img = workspace_images[frame_idx]
                # Handle different image formats
                if workspace_img.dtype != np.uint8:
                    workspace_img = (workspace_img * 255).astype(np.uint8)
                if len(workspace_img.shape) == 3 and workspace_img.shape[2] == 3:
                    # RGB format
                    img = Image.fromarray(workspace_img)
                else:
                    # May need reshaping
                    if workspace_img.shape[0] == 3:
                        workspace_img = workspace_img.transpose(1, 2, 0)
                    img = Image.fromarray(workspace_img)
                
                workspace_img_path = os.path.join(images_dir, f"workspace_{frame_idx:06d}.png")
                img.save(workspace_img_path)
                
                # Save wrist image if available
                if wrist_images is not None and frame_idx < wrist_images.shape[0]:
                    wrist_img = wrist_images[frame_idx]
                    if wrist_img.dtype != np.uint8:
                        wrist_img = (wrist_img * 255).astype(np.uint8)
                    if len(wrist_img.shape) == 3 and wrist_img.shape[2] == 3:
                        wrist_img_pil = Image.fromarray(wrist_img)
                    else:
                        if wrist_img.shape[0] == 3:
                            wrist_img = wrist_img.transpose(1, 2, 0)
                        wrist_img_pil = Image.fromarray(wrist_img)
                    
                    wrist_img_path = os.path.join(images_dir, f"wrist_{frame_idx:06d}.png")
                    wrist_img_pil.save(wrist_img_path)
            
            # Save actions as parquet
            actions_df = pd.DataFrame(
                actions,
                columns=['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
            )
            actions_path = os.path.join(episode_dir, "actions.parquet")
            actions_df.to_parquet(actions_path, index=False)
            
            # Save language task description
            lang_path = os.path.join(episode_dir, "language_task.txt")
            with open(lang_path, 'w') as f:
                f.write(task_description)
            
            episode_idx += 1
        
        print(f"Converted {episode_idx} episodes to {output_split_dir}")


def convert_libero_10_dataset(
    libero_data_dir: str,
    output_dir: str,
    use_huggingface: bool = False
):
    """
    Convert LIBERO_10 dataset to Atlas format
    
    Args:
        libero_data_dir: Directory containing LIBERO datasets (from download)
        output_dir: Output directory for Atlas format
        use_huggingface: Whether data was downloaded from HuggingFace
    """
    # Get LIBERO benchmark
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        libero_10_benchmark = benchmark_dict["libero_10"]()
    except Exception as e:
        print(f"Error loading benchmark: {e}")
        return
    
    # Get default LIBERO data path if not provided
    if libero_data_dir is None:
        libero_data_dir = get_libero_path("datasets")
    
    print(f"LIBERO data directory: {libero_data_dir}")
    print(f"Output directory: {output_dir}")
    
    # LIBERO_10 has 10 tasks
    n_tasks = libero_10_benchmark.n_tasks
    print(f"LIBERO_10 contains {n_tasks} tasks")
    
    # Convert each task
    for task_id in range(n_tasks):
        task_name = libero_10_benchmark.get_task_names()[task_id]
        task_description = libero_10_benchmark.get_task(task_id).language
        
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task_name}")
        print(f"Description: {task_description}")
        print(f"{'='*60}")
        
        # Get demonstration file path
        demo_file = libero_10_benchmark.get_task_demonstration(task_id)
        hdf5_path = os.path.join(libero_data_dir, demo_file)
        
        if not os.path.exists(hdf5_path):
            print(f"Warning: HDF5 file not found: {hdf5_path}")
            print(f"  Expected path: {hdf5_path}")
            print(f"  Please download LIBERO_100 dataset first:")
            print(f"    cd dataset/LIBERO")
            print(f"    python benchmark_scripts/download_libero_datasets.py --datasets libero_100 --use-huggingface")
            continue
        
        # Convert this task
        convert_hdf5_to_atlas_format(
            hdf5_path=hdf5_path,
            output_dir=output_dir,
            benchmark_name="libero_10",
            task_id=task_id,
            split="train"
        )
    
    print(f"\n{'='*60}")
    print(f"Conversion complete! Atlas format data saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO dataset to Atlas format"
    )
    parser.add_argument(
        "--libero-data-dir",
        type=str,
        default=None,
        help="Directory containing LIBERO datasets (default: LIBERO default path)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dataset/libero_10_atlas_format",
        help="Output directory for Atlas format data"
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Data was downloaded from HuggingFace"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_90", "libero_100"],
        help="Which benchmark to convert"
    )
    
    args = parser.parse_args()
    
    if args.benchmark == "libero_10":
        convert_libero_10_dataset(
            libero_data_dir=args.libero_data_dir,
            output_dir=args.output_dir,
            use_huggingface=args.use_huggingface
        )
    else:
        print(f"Conversion for {args.benchmark} not yet implemented")
        print("Currently only libero_10 is supported")


if __name__ == "__main__":
    main()
