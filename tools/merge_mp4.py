#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Optional
import imageio


def find_mp4_files(input_pattern: str, start_index: int = 1, num_files: Optional[int] = None) -> List[Path]:
    input_path = Path(input_pattern)
    
    if input_path.is_file() and input_path.suffix.lower() == '.mp4':
        return [input_path]
    
    if '{' in input_pattern:
        mp4_files = []
        file_idx = start_index
        while True:
            file_path = Path(input_pattern.format(file_idx))
            if not file_path.exists():
                break
            mp4_files.append(file_path)
            file_idx += 1
            if num_files and len(mp4_files) >= num_files:
                break
        return mp4_files
    
    if input_path.is_dir():
        mp4_files = sorted(input_path.glob('*.mp4'))
    else:
        input_dir = input_path.parent if input_path.parent != Path('.') else Path('.')
        pattern = input_path.name if '*' in input_path.name else f"{input_path.name}*.mp4"
        mp4_files = sorted(input_dir.glob(pattern))
    
    if num_files:
        mp4_files = mp4_files[:num_files]
    
    return mp4_files


def merge_mp4_files(mp4_files: List[Path], output_path: Path, fps: float = 30.0, force: bool = False) -> bool:
    if not force and output_path.exists():
        print(f"Output file exists, skipping: {output_path}")
        return False
    
    if not mp4_files:
        raise ValueError("No MP4 files to merge")
    
    all_frames = []
    for mp4_file in mp4_files:
        reader = imageio.get_reader(str(mp4_file))
        for frame in reader:
            all_frames.append(frame)
        reader.close()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_path), all_frames, fps=fps, codec='libx264', quality=8)
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple MP4 files into a single MP4")
    
    parser.add_argument("input", type=str, help="Input directory, file pattern, or single MP4 file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output MP4 file path")
    parser.add_argument("--start-index", type=int, default=1, help="Starting file index for pattern matching (default: 1)")
    parser.add_argument("--num-files", type=int, default=None, help="Maximum number of MP4 files to merge")
    parser.add_argument("--fps", type=float, default=30.0, help="Output frames per second (default: 30.0)")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing output file")
    
    args = parser.parse_args()
    
    mp4_files = find_mp4_files(args.input, start_index=args.start_index, num_files=args.num_files)
    
    if not mp4_files:
        print(f"Error: No MP4 files found matching: {args.input}")
        return
    
    print(f"Found {len(mp4_files)} MP4 files to merge")
    
    output_path = Path(args.output)
    if merge_mp4_files(mp4_files, output_path, fps=args.fps, force=args.force):
        print(f"Successfully merged {len(mp4_files)} files into {output_path}")


if __name__ == "__main__":
    main()