# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tool for creating ZIP/PNG based datasets."""

from torch.utils.data import Dataset, DataLoader
from collections.abc import Iterator
from dataclasses import dataclass
import functools
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from torch_utils import distributed as dist
from training.encoders import StabilityVAEEncoder

PIL.Image.init()
# ----------------------------------------------------------------------------


@dataclass
class ImageEntry:
    img: np.ndarray
    label: Optional[int]

# ----------------------------------------------------------------------------
# Parse a 'M,N' or 'MxN' integer tuple.
# Example: '4x2' returns (4,2)


def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

# ----------------------------------------------------------------------------


def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

# ----------------------------------------------------------------------------


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

# ----------------------------------------------------------------------------


def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

# ----------------------------------------------------------------------------


def open_image_folder(source_dir, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    input_images = []

    def _recurse_dirs(root: str):  # workaround Path().rglob() slowness
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    input_images.append(os.path.join(root, e.name))
                elif e.is_dir():
                    _recurse_dirs(os.path.join(root, e.name))
    _recurse_dirs(source_dir)
    input_images = sorted([f for f in input_images if is_image_ext(f)])

    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)

    # Load labels.
    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    # No labels available => determine from top-level directory names.
    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        for idx, fname in enumerate(input_images):
            img = np.array(PIL.Image.open(fname).convert('RGB'))
            yield ImageEntry(img=img, label=labels.get(arch_fnames[fname]))
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()

# ----------------------------------------------------------------------------


def open_image_zip(source, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)

        # Load labels.
        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = np.array(PIL.Image.open(file).convert('RGB'))
                yield ImageEntry(img=img, label=labels.get(fname))
                if idx >= max_idx - 1:
                    break
    return max_idx, iterate_images()

# ----------------------------------------------------------------------------


def scale(img, w, h):
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return np.array(PIL.Image.fromarray(img).resize((w or img.shape[1], h or img.shape[0]), PIL.Image.Resampling.LANCZOS))


def center_crop(img, w, h):
    c = min(img.shape[:2])
    img = img[(img.shape[0]-c)//2:(img.shape[0]+c)//2, (img.shape[1]-c)//2:(img.shape[1]+c)//2]
    return np.array(PIL.Image.fromarray(img).resize((w, h), PIL.Image.Resampling.LANCZOS))


def center_crop_wide(img, w, h):
    ch = int(round(w * img.shape[0] / img.shape[1]))
    if img.shape[1] < w or ch < h:
        return None
    img = img[(img.shape[0]-ch)//2:(img.shape[0]+ch)//2]
    img = np.array(PIL.Image.fromarray(img).resize((w, h), PIL.Image.Resampling.LANCZOS))
    canvas = np.zeros([w, w, 3], dtype=np.uint8)
    canvas[(w - h)//2:(w + h)//2, :] = img
    return canvas


def center_crop_imagenet(img, size):
    p = PIL.Image.fromarray(img)
    while min(*p.size) >= 2 * size:
        p = p.resize((p.size[0]//2, p.size[1]//2), PIL.Image.Resampling.BOX)
    scale = size / min(*p.size)
    p = p.resize((round(p.size[0]*scale), round(p.size[1]*scale)), PIL.Image.Resampling.BICUBIC)
    arr = np.array(p)
    y, x = (arr.shape[0]-size)//2, (arr.shape[1]-size)//2
    return arr[y:y+size, x:x+size]


class TransformWrapper:
    def __init__(self, mode, w, h):
        self.mode = mode
        self.w = w
        self.h = h

    def __call__(self, img):
        if self.mode is None:
            return scale(img, self.w, self.h)
        if self.mode == 'center-crop':
            return center_crop(img, self.w, self.h)
        if self.mode == 'center-crop-wide':
            return center_crop_wide(img, self.w, self.h)
        if self.mode == 'center-crop-dhariwal':
            return center_crop_imagenet(img, self.w)
        raise ValueError('Unknown transform: ' + self.mode)


def make_transform(transform, w, h):
    return TransformWrapper(transform, w, h)


# ----------------------------------------------------------------------------


def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            raise click.ClickException(f'Only zip archives are supported: {source}')
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

# ----------------------------------------------------------------------------


def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)

        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            raise click.ClickException('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

# ----------------------------------------------------------------------------


@click.group()
def cmdline():
    '''Dataset processing tool for dataset image data conversion and VAE encode/decode preprocessing.'''
    # if os.environ.get('WORLD_SIZE', '1') != '1':
    #     raise click.ClickException('Distributed execution is not supported.')

# ----------------------------------------------------------------------------


@cmdline.command()
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE',            type=click.Choice(['center-crop', 'center-crop-wide', 'center-crop-dhariwal']))
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH',  type=parse_tuple)
def convert(
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]]
):
    """Convert an image dataset into archive format for training.

    Specifying the input images:

    \b
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, class labels are determined from
    top-level directory names.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    The --transform=center-crop-dhariwal selects a crop/rescale mode that is intended
    to exactly match with results obtained for ImageNet in common diffusion model literature:

    \b
    python dataset_tool.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \\
        --dest=datasets/img64.zip --resolution=64x64 --transform=center-crop-dhariwal
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    transform_image = make_transform(transform, *resolution if resolution is not None else (None, None))
    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Apply crop and resize.
        img = transform_image(image.img)
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        assert img.ndim == 3
        cur_image_attrs = {'width': img.shape[1], 'height': img.shape[0]}
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                raise click.ClickException(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if width != 2 ** int(np.floor(np.log2(width))):
                raise click.ClickException('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
            raise click.ClickException(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img)
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image.label] if image.label is not None else None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

# ----------------------------------------------------------------------------


@cmdline.command()
@click.option('--model-url',  help='VAE encoder model', metavar='URL',                  type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--batch', help='Maximum number of images to output', metavar='INT', type=int, default=1)
def encode(
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
    batch: Optional[int],
):
    """Encode pixel data to VAE latents."""
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=batch)
    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    labels = []

    class ImageIterableDataset(torch.utils.data.IterableDataset):
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            for img in self.iterable:
                img_tensor = torch.tensor(img.img).permute(2, 0, 1)  # [C, H, W]
                yield img_tensor, img.label

    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        img_tensor = torch.tensor(image.img).to('cuda').permute(2, 0, 1).unsqueeze(0)
        mean_std = vae.encode_pixels(img_tensor)[0].cpu()
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img-mean-std-{idx_str}.npy'

        f = io.BytesIO()
        np.save(f, mean_std)
        save_bytes(os.path.join(archive_root_dir, archive_fname), f.getvalue())
        labels.append([archive_fname, image.label] if image.label is not None else None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

# ----------------------------------------------------------------------------


@cmdline.command()
@click.option('--model-url',  help='VAE encoder model', metavar='URL',                  type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
def decode(
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
):
    """Decode VAE latents to pixels."""
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=1)
    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    labels = []

    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        std_mean = image.img
        assert isinstance(std_mean, np.ndarray)
        lat = torch.tensor(std_mean).unsqueeze(0).cuda()
        pix = vae.decode(vae.encode_latents(lat))[0].permute(1, 2, 0).cpu().numpy()
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        img = PIL.Image.fromarray(pix, 'RGB')
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image.label] if image.label is not None else None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# python Diffusion_finetuning_project/dataset_tool.py process-and-encode --source=dataset/ --dest=datasetMichel/ --transform=center-crop-dhariwal --batch-size=64 --gpu-batch-size=64 --num-workers=0


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform_fn):
        self.image_paths = image_paths
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = np.array(PIL.Image.open(img_path).convert('RGB'))
        img = self.transform_fn(img)
        if img is None:
            return None
        return img, img_path


@cmdline.command()
@click.option('--source',     help='Input directory containing categorized images', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory for latent embeddings', metavar='PATH',  type=str, required=True)
@click.option('--model-url',  help='VAE encoder model', metavar='URL', type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH', type=parse_tuple, default='512x512')
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE', type=click.Choice(['center-crop', 'center-crop-wide', 'center-crop-dhariwal']))
@click.option('--batch-size', help='Batch size for VAE encoding', metavar='INT', type=int, default=32)
@click.option('--num-workers', help='Number of worker processes', metavar='INT', type=int, default=4)
@click.option('--gpu-batch-size', help='Maximum batch size per GPU', metavar='INT', type=int, default=8)
def process_and_encode(
    source: str,
    dest: str,
    model_url: str,
    resolution: Tuple[int, int],
    transform: Optional[str],
    batch_size: int,
    num_workers: int,
    gpu_batch_size: int,
):
    """Process images and encode them to latents while maintaining folder structure."""

    # import torch.multiprocessing as mp
    # Initialize distributed processing
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create output directory
    os.makedirs(dest, exist_ok=True)

    # Collect all image paths while preserving structure
    image_paths = []
    category_map = {}
    for category in os.listdir(source):
        category_path = os.path.join(source, category)
        if not os.path.isdir(category_path):
            continue

        output_category_path = os.path.join(dest, category)
        os.makedirs(output_category_path, exist_ok=True)

        for img_name in os.listdir(category_path):
            if not is_image_ext(img_name):
                continue
            img_path = os.path.join(category_path, img_name)
            image_paths.append(img_path)
            category_map[img_path] = category

    # Initialize transform function

    transform_fn = make_transform(transform, *resolution)

    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, transform_fn)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    print(f"[Rank {rank}] Processing {len(dataloader)} batches.", flush=True)
    print(f"[Rank {rank}] Total images in shard: {len(sampler)}", flush=True)

    # Initialize VAE encoder
    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=gpu_batch_size)
    vae.init(torch.device(f'cuda:{rank}'))
    # Process batches
    for batch_idx, (images, paths) in tqdm(enumerate(dataloader), total=len(dataloader), disable=dist.get_rank() != 0):
        if images is None:
            continue

        # Process in smaller GPU batches
        latents = []
        images = images.permute(0, 3, 1, 2).requires_grad_(False).to(f'cuda:{rank}')
        latents = vae.encode_pixels(images).cpu()

        for path, latent in zip(paths, latents):
            category = category_map[path]
            latent_name = os.path.splitext(os.path.basename(path))[0] + '.npy'
            latent_path = os.path.join(dest, category, latent_name)
            np.save(latent_path, latent)

    # dist.destroy_process_group()
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    cmdline()

# ----------------------------------------------------------------------------
