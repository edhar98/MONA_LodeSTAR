#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from multiprocessing import Pool, cpu_count
import numpy as np
from nptdms import TdmsFile
from PIL import Image
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def list_tdms_structure(tdms_path: Path) -> str:
    tdms_file = TdmsFile.read(str(tdms_path))
    groups = tdms_file.groups()
    
    if not groups:
        return f"No groups found in {tdms_path}"
    
    lines = [f"TDMS file structure for {tdms_path}:"]
    for group in groups:
        group_path = group.path
        channels = group.channels()
        lines.append(f"  Group: '{group_path}' ({len(channels)} channels)")
        for idx, channel in enumerate(channels):
            try:
                data = channel[:]
                lines.append(f"    Channel {idx}: {channel.name}, shape={data.shape}, dtype={data.dtype}")
            except Exception as e:
                lines.append(f"    Channel {idx}: {channel.name}, error reading data: {e}")
    
    return "\n".join(lines)


def extract_images_from_tdms(
    tdms_path: Path,
    image_width: int = 1024,
    image_height: int = 1024,
    channel_index: int = 0,
    group_name: Optional[str] = None
) -> np.ndarray:
    tdms_file = TdmsFile.read(str(tdms_path))
    groups = tdms_file.groups()
    
    if not groups:
        raise ValueError(f"No groups found in {tdms_path}")
    
    group_paths = [g.path for g in groups]
    
    if group_name is None:
        group = groups[0]
    else:
        matching_groups = [g for g in groups if g.path == group_name]
        if not matching_groups:
            available = ", ".join(f"'{g}'" for g in group_paths)
            structure = list_tdms_structure(tdms_path)
            raise ValueError(
                f"Group '{group_name}' not found in {tdms_path}\n"
                f"Available groups: {available}\n\n{structure}"
            )
        group = matching_groups[0]
    
    channels = group.channels()
    
    if channel_index >= len(channels):
        raise ValueError(f"Channel index {channel_index} out of range. Found {len(channels)} channels")
    
    channel = channels[channel_index]
    image_data = channel[:]
    
    image_size = image_width * image_height
    total_pixels = image_data.size
    
    if total_pixels % image_size != 0:
        raise ValueError(f"Data size {total_pixels} is not divisible by image size {image_size}")
    
    num_images = total_pixels // image_size
    images = image_data.reshape(num_images, image_height, image_width)
    
    return images


def save_images(
    images: np.ndarray,
    output_dir: Path,
    base_name: str,
    start_index: int = 1,
    dtype: Optional[np.dtype] = None,
    force: bool = False
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    source_dtype = images.dtype
    
    if dtype is None:
        dtype = source_dtype
    
    saved_count = 0
    skipped_count = 0
    for idx, image in enumerate(images):
        output_path = output_dir / f"{base_name}_{start_index + idx:03d}.png"
        
        if not force and output_path.exists():
            skipped_count += 1
            continue
        
        if dtype == source_dtype:
            image_converted = image
        elif dtype == np.uint8:
            if source_dtype == np.uint16:
                image_converted = (image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
            elif source_dtype in (np.float32, np.float64):
                image_min = image.min()
                image_max = image.max()
                if image_max > image_min:
                    image_converted = ((image - image_min) / (image_max - image_min) * 255.0).astype(np.uint8)
                else:
                    image_converted = np.zeros_like(image, dtype=np.uint8)
            else:
                image_converted = np.clip(image, 0, 255).astype(np.uint8)
        elif dtype == np.uint16:
            if source_dtype == np.uint8:
                image_converted = (image.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
            elif source_dtype in (np.float32, np.float64):
                image_min = image.min()
                image_max = image.max()
                if image_max > image_min:
                    image_converted = ((image - image_min) / (image_max - image_min) * 65535.0).astype(np.uint16)
                else:
                    image_converted = np.zeros_like(image, dtype=np.uint16)
            else:
                image_converted = np.clip(image, 0, 65535).astype(np.uint16)
        else:
            image_converted = image.astype(dtype)
        
        if dtype == np.uint8:
            img = Image.fromarray(image_converted, mode='L')
        elif dtype == np.uint16:
            img = Image.fromarray(image_converted, mode='I;16')
        else:
            img = Image.fromarray(image_converted)
        
        img.save(output_path)
        saved_count += 1
    
    return saved_count


def save_video(
    images: np.ndarray,
    output_path: Path,
    fps: float = 30.0,
    dtype: Optional[np.dtype] = None,
    force: bool = False
) -> bool:
    if not HAS_IMAGEIO:
        raise ImportError("imageio is required for video conversion. Install it with: pip install imageio imageio-ffmpeg")
    
    if not force and output_path.exists():
        return False
    
    source_dtype = images.dtype
    
    if dtype is None:
        dtype = source_dtype
    
    video_frames = []
    for image in images:
        if dtype == source_dtype:
            image_converted = image
        elif dtype == np.uint8:
            if source_dtype == np.uint16:
                image_converted = (image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
            elif source_dtype in (np.float32, np.float64):
                image_min = image.min()
                image_max = image.max()
                if image_max > image_min:
                    image_converted = ((image - image_min) / (image_max - image_min) * 255.0).astype(np.uint8)
                else:
                    image_converted = np.zeros_like(image, dtype=np.uint8)
            else:
                image_converted = np.clip(image, 0, 255).astype(np.uint8)
        elif dtype == np.uint16:
            if source_dtype == np.uint8:
                image_converted = (image.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
            elif source_dtype in (np.float32, np.float64):
                image_min = image.min()
                image_max = image.max()
                if image_max > image_min:
                    image_converted = ((image - image_min) / (image_max - image_min) * 65535.0).astype(np.uint16)
                else:
                    image_converted = np.zeros_like(image, dtype=np.uint16)
            else:
                image_converted = np.clip(image, 0, 65535).astype(np.uint16)
        else:
            image_converted = image.astype(dtype)
        
        if dtype == np.uint8:
            video_frames.append(image_converted)
        elif dtype == np.uint16:
            image_converted_uint8 = (image_converted.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
            video_frames.append(image_converted_uint8)
        else:
            image_min = image_converted.min()
            image_max = image_converted.max()
            if image_max > image_min:
                image_converted_uint8 = ((image_converted - image_min) / (image_max - image_min) * 255.0).astype(np.uint8)
            else:
                image_converted_uint8 = np.zeros_like(image_converted, dtype=np.uint8)
            video_frames.append(image_converted_uint8)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    imageio.mimwrite(
        str(output_path),
        video_frames,
        fps=fps,
        codec='libx264',
        quality=8
    )
    
    return True


def process_single_tdms_file(
    args: Tuple[Path, Path, int, int, int, int, Optional[str], Optional[str], Optional[np.dtype], bool, float, bool]
) -> int:
    tdms_file, output_dir, start_index, image_width, image_height, channel_index, group_name, base_name, dtype, to_video, fps, force = args
    
    images = extract_images_from_tdms(
        tdms_file,
        image_width=image_width,
        image_height=image_height,
        channel_index=channel_index,
        group_name=group_name
    )
    
    file_base_name = base_name
    if file_base_name is None:
        file_base_name = tdms_file.stem.replace('_video', '')
    
    saved = save_images(
        images,
        output_dir,
        file_base_name,
        start_index=start_index,
        dtype=dtype,
        force=force
    )
    
    if to_video:
        output_path = output_dir / f"{file_base_name}.mp4"
        save_video(images, output_path, fps=fps, dtype=dtype, force=force)
    
    return saved


def process_tdms_files(
    input_pattern: str,
    output_dir: Path,
    start_index: int = 1,
    num_files: Optional[int] = None,
    image_width: int = 1024,
    image_height: int = 1024,
    channel_index: int = 0,
    group_name: Optional[str] = None,
    base_name: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
    num_workers: Optional[int] = None,
    to_video: bool = False,
    fps: float = 30.0,
    force: bool = False
) -> int:
    input_path = Path(input_pattern)
    
    if input_path.is_file():
        tdms_files = [input_path]
    elif '{' in input_pattern:
        tdms_files = []
        file_idx = start_index
        while True:
            file_path = Path(input_pattern.format(file_idx))
            if not file_path.exists():
                break
            tdms_files.append(file_path)
            file_idx += 1
            if num_files and len(tdms_files) >= num_files:
                break
    else:
        input_dir = input_path.parent if input_path.parent != Path('.') else Path('.')
        pattern = input_path.name if input_path.name != input_pattern else '*'
        tdms_files = sorted(input_dir.glob(f"{pattern}.tdms"))
        if num_files:
            tdms_files = tdms_files[:num_files]
    
    if not tdms_files:
        raise FileNotFoundError(f"No TDMS files found matching pattern: {input_pattern}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if num_workers is None:
        num_workers = cpu_count()
    
    if num_workers == 1 or len(tdms_files) == 1:
        total_saved = 0
        for tdms_file in tdms_files:
            images = extract_images_from_tdms(
                tdms_file,
                image_width=image_width,
                image_height=image_height,
                channel_index=channel_index,
                group_name=group_name
            )
            
            file_base_name = base_name
            if file_base_name is None:
                file_base_name = tdms_file.stem.replace('_video', '')
            
            saved = save_images(
                images,
                output_dir,
                file_base_name,
                start_index=start_index,
                dtype=dtype,
                force=force
            )
            total_saved += saved
            
            if to_video:
                output_path = output_dir / f"{file_base_name}.mp4"
                save_video(images, output_path, fps=fps, dtype=dtype, force=force)
        return total_saved
    
    tasks = [
        (tdms_file, output_dir, start_index, image_width, image_height, channel_index, group_name, base_name, dtype, to_video, fps, force)
        for tdms_file in tdms_files
    ]
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_tdms_file, tasks)
    
    return sum(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TDMS files to PNG images or MP4 videos",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input TDMS file path or pattern (use {:03d} for numbered files)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=False,
        help="Output directory for PNG images (required unless --list-structure)"
    )
    
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting file index for pattern matching (default: 1)"
    )
    
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Maximum number of TDMS files to process (default: all found)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (default: 1024)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels (default: 1024)"
    )
    
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to extract (default: 0)"
    )
    
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="TDMS group name (default: first available group)"
    )
    
    parser.add_argument(
        "--list-structure",
        action="store_true",
        help="List TDMS file structure and exit"
    )
    
    parser.add_argument(
        "--base-name",
        type=str,
        default=None,
        help="Base name for output PNG files (default: derived from input)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["uint8", "uint16", "float32"],
        default=None,
        help="Output data type (default: preserve input type)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: number of CPU cores, {cpu_count()})"
    )
    
    parser.add_argument(
        "--to-mp4",
        action="store_true",
        help="Also create MP4 videos in addition to PNG images"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for video output (default: 30.0)"
    )
    
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing output files (default: skip existing files)"
    )
    
    args = parser.parse_args()
    
    dtype_map = {
        "uint8": np.uint8,
        "uint16": np.uint16,
        "float32": np.float32
    }
    
    dtype = dtype_map.get(args.dtype) if args.dtype else None
    
    try:
        if args.list_structure:
            input_path = Path(args.input)
            if '{' in args.input:
                input_path = Path(args.input.format(args.start_index))
            
            if not input_path.exists():
                print(f"Error: File not found: {input_path}", file=os.sys.stderr)
                os.sys.exit(1)
            
            print(list_tdms_structure(input_path))
            return
        
        if not args.output:
            parser.error("--output is required unless using --list-structure")
        
        if args.to_mp4 and not HAS_IMAGEIO:
            print("Error: imageio is required for video conversion. Install it with: pip install imageio imageio-ffmpeg", file=os.sys.stderr)
            os.sys.exit(1)
        
        total_saved = process_tdms_files(
            input_pattern=args.input,
            output_dir=Path(args.output),
            start_index=args.start_index,
            num_files=args.num_files,
            image_width=args.width,
            image_height=args.height,
            channel_index=args.channel,
            group_name=args.group,
            base_name=args.base_name,
            dtype=dtype,
            num_workers=args.workers,
            to_video=args.to_mp4,
            fps=args.fps,
            force=args.force
        )
        
        if args.to_mp4:
            num_videos = len([f for f in Path(args.output).glob('*.mp4')])
            print(f"Successfully converted {total_saved} images and {num_videos} video(s) to {args.output}")
        else:
            print(f"Successfully converted {total_saved} images to {args.output}")
    
    except Exception as e:
        print(f"Error: {e}", file=os.sys.stderr)
        os.sys.exit(1)


if __name__ == "__main__":
    main()
