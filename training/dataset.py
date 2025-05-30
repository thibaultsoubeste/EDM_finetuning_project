# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from torch_utils.distributed import print0

try:
    import pyspng
except ImportError:
    pyspng = None

# ---------------------------------------------------------------------------
# Preprocessing if needed:


class CenterCropImagenet:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if img.shape[1:] == (self.size, self.size):
            return img
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        if img.shape[2] == 1:
            img = img[:, :, 0]
        p = PIL.Image.fromarray(img).convert('RGB')
        while min(*p.size) >= 2 * self.size:
            p = p.resize((p.size[0]//2, p.size[1]//2), PIL.Image.Resampling.BOX)
        scale = self.size / min(*p.size)
        p = p.resize((round(p.size[0]*scale), round(p.size[1]*scale)), PIL.Image.Resampling.BICUBIC)
        arr = np.array(p)
        y, x = (arr.shape[0] - self.size) // 2, (arr.shape[1] - self.size) // 2
        return np.transpose(arr[y:y+self.size, x:x+self.size], (2, 0, 1))

# ----------------------------------------------------------------------------
# Abstract base class for datasets.


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 name,                   # Name of the dataset.
                 raw_shape,              # Shape of the raw image data (NCHW).
                 use_labels=True,     # Enable conditioning labels? False = label dimension is zero.
                 max_size=None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 xflip=False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 random_seed=0,        # Random seed to use when applying max_size.
                 cache=False,    # Cache images in CPU memory?
                 preprocess=None,
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict()  # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None
        self._preprocess = preprocess

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._preprocess is not None:
                image = self._preprocess(image)

            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self._raw_shape[1:], (f'image shape={list(image.shape)}, expected={self._raw_shape[1:]}, {idx=}, image name {self._image_fnames[raw_idx]}'
                                                          )
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):  # [CHW]
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

# ----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.


class ImageFolderDataset(Dataset):
    def __init__(self,
                 path,                   # Path to directory or zip.
                 resolution=None,        # Ensure specific resolution, None = anything goes.
                 **super_kwargs,         # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        preprocess = None
        if resolution is not None:
            preprocess = CenterCropImagenet(resolution)
            raw_shape = [len(self._image_fnames), 3, resolution, resolution]
            # raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, preprocess=preprocess, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        ext = self._file_ext(fname)
        with self._open_file(fname) as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                try:
                    image = np.array(PIL.Image.open(f).convert('RGB'))
                except Exception as e:
                    print0(f"Skipping corrupted file: {fname} ({e})")
                    return np.zeros((self._raw_shape[1:]), dtype=np.uint8)
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


# ----------------------------------------------------------------------------

class FilteredImageDataset(ImageFolderDataset):
    def __init__(self,
                 path,                   # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = anything goes.
                 nima_threshold=None,  # Minimum NIMA score to include (None = no filtering)
                 categories=None,  # List of categories to include (None = all categories)
                 top_percent=None,  # Keep top X% of images by NIMA score (None = no filtering)
                 top_per_category=None,  # Keep top X% of images in each category (None = no filtering)
                 max_size=None,
                 **super_kwargs,         # Additional arguments for the Dataset base class.
                 ):
        if path.endswith('.zip'):
            raise NotImplementedError("Custom dataloader doesn't work with zip format")

        # Load and filter metadata
        super().__init__(path=path, resolution=resolution, **super_kwargs)
        self._max_size = max_size
        self._metadata = self._load_metadata()
        self._uniquelabels = np.sort([folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))])
        self._labels2idxmapping = {foldername: idx for idx, foldername in enumerate(self._uniquelabels)}
        self._extenstion = os.path.splitext(self._image_fnames[0])[1]
        if nima_threshold is not None or categories is not None or top_percent is not None or top_per_category is not None:
            self._filter_fnames(nima_threshold, categories, top_percent, top_per_category)

    def _load_metadata(self):
        """Load metadata from dataset.json with caching."""
        if not hasattr(self, '_metadata_cache'):
            meta_fname = 'metadata.json'
            if meta_fname not in self._all_fnames:
                return {}
            with self._open_file(os.path.abspath(os.path.join(self._path, meta_fname))) as f:
                data = json.load(f)
                self._metadata_cache = data.get('metadata', {})
        return self._metadata_cache

    def _filter_fnames(self, nima_threshold, categories, top_percent, top_per_category):
        """Filter image filenames based on NIMA score, categories, and top percentage."""
        if not self._metadata:
            return
        categories = self._uniquelabels[categories]
        # First filter by NIMA threshold and categories
        filtered_fnames = []
        category_scores = {}  # Store scores by category for later filtering

        for fname in self._image_fnames:
            rel_path = fname.replace('\\', '/')
            meta = self._metadata.get(rel_path.replace(self._extenstion, ''), {})
            # Check NIMA score
            if nima_threshold is not None:
                nima_score = meta.get('nima', 0)
                if nima_score < nima_threshold:
                    continue

            # Check category
            category = meta.get('labels', '')
            if categories is not None and category not in categories:
                continue
            # Store scores for later filtering
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append((fname, meta.get('nima', 0)))
            filtered_fnames.append(fname)

        # Apply top percentage filtering if requested
        if top_percent is not None or top_per_category is not None:
            final_fnames = set()

            # Filter by top percentage per category
            if top_per_category is not None:
                for category, scores in category_scores.items():
                    if not scores:
                        continue
                    # Sort by NIMA score in descending order
                    scores.sort(key=lambda x: x[1], reverse=True)
                    # Calculate number of images to keep (at least 1)
                    keep_count = max(1, int(len(scores) * top_per_category / 100))
                    # Add top images to final set
                    final_fnames.update(fname for fname, _ in scores[:keep_count])

            # Filter by overall top percentage
            if top_percent is not None:
                # Combine all scores
                all_scores = [(fname, meta.get('nima', 0)) for fname in filtered_fnames]
                # Sort by NIMA score in descending order
                all_scores.sort(key=lambda x: x[1], reverse=True)
                # Calculate number of images to keep
                keep_count = int(len(all_scores) * top_percent / 100)
                # Add top images to final set
                final_fnames.update(fname for fname, _ in all_scores[:keep_count])

            # Update filtered_fnames with the intersection of both filters
            if top_percent is not None and top_per_category is not None:
                filtered_fnames = list(final_fnames)
            elif top_percent is not None:
                filtered_fnames = [fname for fname, _ in all_scores[:keep_count]]
            elif top_per_category is not None:
                filtered_fnames = list(final_fnames)

        self._image_fnames = filtered_fnames
        self._raw_shape[0] = len(filtered_fnames)
        print0(f'After filtration there are {self._raw_shape[0]} images in the dataset')

        # Update raw_idx to match new filtered size
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if self._max_size is not None and self._raw_idx.size > self._max_size:
            np.random.RandomState(self._random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:self._max_size])

        # Update xflip if needed
        if self._xflip.sum() > 0:
            self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])
        else:
            self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)

    def get_metadata(self, idx):
        """Get metadata for a specific image index."""
        fname = self._image_fnames[self._raw_idx[idx]]
        rel_path = fname.replace('\\', '/')
        return self._metadata.get(rel_path, {})

    def _load_raw_labels(self):
        labels = {k: v["labels"] for k, v in self._metadata.items()}
        if not labels:
            return None

        # Match filenames
        labels = [labels[fname.replace('\\', '/').replace(self._extenstion, '')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = np.vectorize(lambda x: self._labels2idxmapping[x])(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
