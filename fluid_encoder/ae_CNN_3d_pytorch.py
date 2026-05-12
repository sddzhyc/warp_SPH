#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple 3D convolutional autoencoder for fluid velocity fields.

This script mirrors the 2D reference implementation and adds standalone
training/testing entry points that read the default fluid-field dataset.
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "fluid_grid_dataset")
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from encoder_utils import normalize_fluid_field
except ModuleNotFoundError:
    from fluid_encoder.encoder_utils import normalize_fluid_field


class AutoEncoder3D(nn.Module):
    def __init__(self, in_channels=3, latent_channels=128, latent_dim=128, input_grid_shape=(64, 32, 32)):
        super(AutoEncoder3D, self).__init__()
        self.latent_channels = int(latent_channels)
        self.latent_dim = int(latent_dim)

        d, h, w = tuple(int(v) for v in input_grid_shape)
        self._conv_out_shape = (d // 8, h // 8, w // 8)
        if min(self._conv_out_shape) <= 0:
            raise ValueError(f"input_grid_shape is too small for 3 downsampling stages: {input_grid_shape}")
        self._conv_flat_dim = self.latent_channels * self._conv_out_shape[0] * self._conv_out_shape[1] * self._conv_out_shape[2]

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),

            nn.Conv3d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),

            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),

            nn.Flatten(start_dim=1),
            nn.Linear(self._conv_flat_dim, self.latent_dim),
            nn.LeakyReLU(0.1),

        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self._conv_flat_dim),
            nn.LeakyReLU(0.1),
            nn.Unflatten(
                dim=1,
                unflattened_size=(
                    self.latent_channels,
                    self._conv_out_shape[0],
                    self._conv_out_shape[1],
                    self._conv_out_shape[2],
                ),
            ),

            nn.ConvTranspose3d(self.latent_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose3d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose3d(64, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class FluidFieldDataset(Dataset):
    def __init__(self, file_paths, normalize=False):
        self.file_paths = list(file_paths)
        self.normalize = bool(normalize)

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def _load_velocity_field(file_path):
        with np.load(file_path) as data:
            if "velocity_field" in data:
                field = data["velocity_field"]
            else:
                field = data[data.files[0]]

        field = np.asarray(field, dtype=np.float32)
        if field.ndim == 4 and field.shape[0] != 3 and field.shape[-1] == 3:
            field = np.transpose(field, (3, 0, 1, 2))
        if field.ndim != 4 or field.shape[0] != 3:
            raise ValueError(f"Expected velocity_field with shape (3, D, H, W), got {field.shape} from {file_path}")
        return field

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        field = self._load_velocity_field(file_path)
        if self.normalize:
            # Normalize a single 3D sample with an explicit batch axis so the
            # spatial dimensions are treated consistently with batched inputs.
            field = normalize_fluid_field(field[None, ...])[0].astype(np.float32)
        return torch.from_numpy(field)


def _infer_velocity_field_shape(file_paths):
    if not file_paths:
        raise ValueError("Cannot infer velocity-field shape from an empty file list")

    first_field = FluidFieldDataset._load_velocity_field(file_paths[0])
    return tuple(int(v) for v in first_field.shape[1:])


def _list_dataset_files(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not file_paths:
        raise FileNotFoundError(f"No .npz files found in: {data_dir}")
    return file_paths


def _split_file_paths(file_paths, train_ratio, seed):
    if len(file_paths) == 1:
        return list(file_paths), list(file_paths)

    generator = torch.Generator().manual_seed(int(seed))
    train_size = max(1, int(round(len(file_paths) * float(train_ratio))))
    train_size = min(train_size, len(file_paths) - 1)
    test_size = len(file_paths) - train_size
    train_subset, test_subset = random_split(file_paths, [train_size, test_size], generator=generator)
    train_files = [file_paths[i] for i in train_subset.indices]
    test_files = [file_paths[i] for i in test_subset.indices]
    return train_files, test_files


def _make_loader(file_paths, batch_size, shuffle, normalize, num_workers):
    dataset = FluidFieldDataset(file_paths, normalize=normalize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def _load_checkpoint(model, ckpt_path, device):
    if not ckpt_path:
        raise ValueError("ckpt_path must be provided for testing")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    return ckpt_path


def train_autoencoder(args):
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    file_paths = _list_dataset_files(args.data_dir)
    train_files, test_files = _split_file_paths(file_paths, args.train_ratio, args.seed)
    input_grid_shape = _infer_velocity_field_shape(train_files)
    print(f"Using velocity-field grid shape: {input_grid_shape}")

    train_loader = _make_loader(train_files, args.batch_size, shuffle=True, normalize=args.normalize, num_workers=args.num_workers)
    test_loader = _make_loader(test_files, args.batch_size, shuffle=False, normalize=args.normalize, num_workers=args.num_workers)

    model = AutoEncoder3D(in_channels=3, input_grid_shape=input_grid_shape).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_base = os.path.join(args.save_dir, start_time)
    os.makedirs(models_base, exist_ok=True)

    best_test_loss = float("inf")
    best_ckpt_path = None

    print(f"Using dataset: {args.data_dir}")
    print(f"Train files: {len(train_files)} | Test files: {len(test_files)}")
    print(f"Saving checkpoints to: {models_base}")

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_num = 0

        for batch in train_loader:
            batch = batch.to(device=device, dtype=torch.float32)
            _, recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_num += batch_size

        train_loss = train_loss_sum / max(1, train_num)
        test_loss = evaluate_autoencoder(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch:04d} | train_loss={train_loss:.6e} | test_loss={test_loss:.6e} | "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        epoch_ckpt = os.path.join(models_base, f"autoencoder_epoch{epoch:04d}.pth")
        torch.save(model.state_dict(), epoch_ckpt)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_ckpt_path = os.path.join(models_base, f"autoencoder_best_test_loss_{best_test_loss:.6e}.pth")
            torch.save(model.state_dict(), best_ckpt_path)

    last_ckpt_path = os.path.join(models_base, f"autoencoder_last_test_loss_{best_test_loss:.6e}.pth")
    torch.save(model.state_dict(), last_ckpt_path)

    print(f"Saved last checkpoint to: {last_ckpt_path}")
    if best_ckpt_path is not None:
        print(f"Saved best checkpoint to: {best_ckpt_path}")


@torch.no_grad()
def evaluate_autoencoder(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_num = 0

    for batch in data_loader:
        batch = batch.to(device=device, dtype=torch.float32)
        _, recon = model(batch)
        loss = criterion(recon, batch)

        batch_size = batch.size(0)
        total_loss += float(loss.item()) * batch_size
        total_num += batch_size

    return total_loss / max(1, total_num)


def test_autoencoder(args):
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    file_paths = _list_dataset_files(args.data_dir)
    _, test_files = _split_file_paths(file_paths, args.train_ratio, args.seed)
    input_grid_shape = _infer_velocity_field_shape(test_files)
    print(f"Using velocity-field grid shape: {input_grid_shape}")
    test_loader = _make_loader(test_files, args.batch_size, shuffle=False, normalize=args.normalize, num_workers=args.num_workers)

    model = AutoEncoder3D(
        in_channels=3,
        latent_channels=args.latent_channels,
        input_grid_shape=input_grid_shape,
    ).to(device)
    _load_checkpoint(model, args.ckpt_path, device)
    criterion = nn.MSELoss()

    test_loss = evaluate_autoencoder(model, test_loader, criterion, device)
    print(f"Test dataset: {args.data_dir}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Test files: {len(test_files)}")
    print(f"Test loss: {test_loss:.6e}")


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing fluid-field .npz files.")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory used to save checkpoints.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path used for testing.")
    parser.add_argument("--train", action="store_true", help="Train the autoencoder.")
    parser.add_argument("--test_only", action="store_true", help="Evaluate the autoencoder on the test split.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Fraction of files used for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for dataset split.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--normalize", action="store_true", help="Apply fluid-field normalization before training/testing.")
    return parser


def main():
    args = get_parser().parse_args()

    if not args.train and not args.test_only:
        args.test_only = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.train:
        train_autoencoder(args)

    if args.test_only:
        test_autoencoder(args)


if __name__ == '__main__':
    main()