from calculate_metrics import get_detector
from torch import nn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc
from training import dataset
import generate_images
import json
import glob
import pandas as pd


class CheckpointDataset(dataset.ImageFolderDataset):
    def __init__(self, path, checkpoint_path, resolution=None, **super_kwargs):
        super().__init__(path, resolution, **super_kwargs)

        processed = set()
        self.checkpoint_nb = 0
        for f in glob.glob(os.path.join(checkpoint_path, "results_*.csv")):
            df = pd.read_csv(f)
            processed.update(df['filename'].tolist())
            self.checkpoint_nb += 1

        if len(processed) != 0:
            dist.print0('Retrieving previous checkpoint')
            self._image_fnames = [f for f in tqdm.tqdm(self._image_fnames, disable=(dist.get_rank() != 0)) if os.path.splitext(f)[0].replace(os.sep, '/') not in processed]
            self._raw_shape[0] = len(self._image_fnames)
            self._raw_idx = np.arange(len(self._image_fnames), dtype=np.int64)

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)  # ignore label
        raw_idx = self._raw_idx[idx]
        fname = self._image_fnames[raw_idx]
        return image, fname


def score_dataset(
    image_path,             # Path to a directory or ZIP file containing the images.
    out_path,               # output path
    metric,                 # Model to use
    max_batch_size=64,   # Maximum batch size.
    num_workers=2,    # How many subprocesses to use for data loading.
    prefetch_factor=2,    # Number of images loaded in advance by each worker.
    verbose=True,  # Enable status prints?
    resolution=512,  # default res
    checkpoints=None
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = CheckpointDataset(path=image_path, checkpoint_path=out_path, resolution=resolution)
    checkpoint_nb = dataset_obj.checkpoint_nb
    if len(dataset_obj) == 0 and checkpoint_nb > 0:
        dist.print0('Dataset Already Fully Processed')
        return

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank():: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
                                              num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))
    device = torch.device('cuda')
    model = get_detector(metric=metric, verbose=True)
    os.makedirs(out_path, exist_ok=True)
    if verbose:
        dist.print0('Scoring Images...')

    # Loop over batches.
    result = {}
    for idx, (images, fnames) in tqdm.tqdm(enumerate(data_loader), unit='batch', disable=(dist.get_rank() != 0), total=len(data_loader)):
        fnames = [os.path.splitext(f)[0].replace(os.sep, '/') for f in fnames]
        images = torch.as_tensor(images).to(device)

        # Accumulate statistics.
        if images is not None:
            scores = model(images).squeeze(1).detach().cpu().tolist()
            result.update(dict(zip(fnames, scores)))
            if checkpoints is not None and (idx+1) % checkpoints == 0:

                if verbose:
                    dist.print0('\nCheckpointing')
                gathered = [None for _ in range(dist.get_world_size())]
                torch.distributed.all_gather_object(gathered, result)

                if dist.get_rank() == 0:
                    merged = {}
                    for partial in gathered:
                        merged.update(partial)
                    pd.Series(merged, name=metric).to_csv(os.path.join(out_path, f'results_{checkpoint_nb}.csv'), header=True, index_label="filename")
                result = {}
                checkpoint_nb += 1

    if verbose:
        dist.print0('\nCheckpointing')
    gathered = [None for _ in range(dist.get_world_size())]
    torch.distributed.all_gather_object(gathered, result)

    if dist.get_rank() == 0:
        merged = {}
        for partial in gathered:
            merged.update(partial)
        pd.Series(merged, name=metric).to_csv(os.path.join(out_path, f'results_{checkpoint_nb}.csv'), header=True, index_label="filename")


@click.command()
@click.option('--images', 'image_path',     help='Path to the images', metavar='PATH|ZIP',                  type=str, required=True)
@click.option('--out', 'out_path',          help='Output file', metavar='PATH',                             type=str, required=True)
@click.option('--metric',                   help='List of metrics to compute', metavar='LIST',              type=str, required=True, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',     type=click.IntRange(min=0), default=2, show_default=True)
@click.option('--checkpoints',              help='Number of classes between checkpoints', metavar='PATH',    type=click.IntRange(min=0), default=None)
def cmdline(image_path, metric, out_path, checkpoints, **opts):
    """Compute a metric over an entire dataset"""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    score_dataset(image_path=image_path,  out_path=out_path, metric=metric, checkpoints=checkpoints, **opts)
    if dist.get_rank() == 0:
        dist.print0('Concatenating the results')
        csvs = sorted(glob.glob(os.path.join(out_path, "results_*.csv")))
        all_scores = pd.concat([pd.read_csv(f).set_index("filename") for f in csvs])
        all_scores.to_csv(os.path.join(out_path, "results_full.csv"), header=True, index_label="filename")
    torch.distributed.barrier()


if __name__ == '__main__':
    cmdline()
