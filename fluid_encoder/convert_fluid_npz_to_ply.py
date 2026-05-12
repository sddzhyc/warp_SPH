#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch convert saved fluid-field NPZ files into Houdini-friendly PLY point clouds."""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fluid_encoder.encoder_utils import batch_convert_fluid_npz_dir_to_houdini_ply


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "datasets", "fluid_grid_dataset"),
        help="Directory containing saved fluid-field .npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for exported .ply files. Defaults to input_dir.",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=0.0,
        help="Discard grid cells whose accumulated weight is below this threshold.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    output_paths = batch_convert_fluid_npz_dir_to_houdini_ply(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_weight=args.min_weight,
    )

    print(f"Converted {len(output_paths)} files")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()