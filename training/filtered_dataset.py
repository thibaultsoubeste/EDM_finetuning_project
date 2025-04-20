import os
import json
import click
import re
import numpy as np
import torch
from . import training_loop
from . import networks
from . import dataset
from . import filtered_dataset
from . import metrics
from . import util

@click.command()
@click.pass_context
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--data', help='Training data', required=True, metavar='[ZIP|DIR]')
@click.option('--nima-threshold', help='Minimum NIMA score for images', type=float, default=None, show_default=True)
@click.option('--categories', help='Comma-separated list of categories to include', type=str, default=None, show_default=True)
@click.option('--top-percent', help='Keep top X% of images by NIMA score', type=float, default=None, show_default=True)
@click.option('--top-per-category', help='Keep top X% of images in each category', type=float, default=None, show_default=True)

def main(ctx, outdir, data, nima_threshold, categories, top_percent, top_per_category, **kwargs):
    # Parse categories if provided
    if categories is not None:
        categories = [cat.strip() for cat in categories.split(',')]
    
    # Initialize dataset
    training_set_kwargs = dnnlib.EasyDict(
        class_name='training.filtered_dataset.FilteredImageDataset',
        path=data,
        nima_threshold=nima_threshold,
        categories=categories,
        top_percent=top_percent,
        top_per_category=top_per_category
    )
    training_set_kwargs.update(kwargs)
    
    # Initialize training loop
    training_loop.training_loop(training_set_kwargs=training_set_kwargs, **kwargs) 
