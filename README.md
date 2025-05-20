# EDM2 and Autoguidance Fork — Efficient Second-Stage Training

**Fork of:** Analyzing and Improving the Training Dynamics of Diffusion Models (CVPR 2024 oral) and Guiding a Diffusion Model with a Bad Version of Itself (NeurIPS 2024 oral)

* Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, Samuli Laine
* [EDM2 paper](https://arxiv.org/abs/2312.02696)
* [Autoguidance paper](https://arxiv.org/abs/2406.02507)

---

## 🔄 About This Fork

This fork focuses on exploring **two-stage training** strategies using the EDM2 architecture. While the first stage uses the original weights, this fork introduces a second stage of fine-tuning with curated data and specialized training enhancements.

This is made possible by the availability of training snapshots across the full training trajectory in the [raw snapshot archive](https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/). One can pick a checkpoint corresponding to an intermediate model and use it as the initialization for further targeted training.

### 🔧 Key Enhancements in This Fork

* **Custom dataset for curated subsets**: Efficient loading and filtering training data based on quality, using internal aesthetic scoring or external filters.
* **Optimized for slow NFS**: Improvements for caching and streaming in distributed environments.
* **Multi-FID / FD Evaluation**: Support multiple reference datasets to assess generation improvements and detect mode collapse
* **Aesthetic scoring support**: Integrated image scorers (e.g., NIMA) for filtering training data by quality.
* **Architecture freezing**: Optional layer freezing for targeted adaptation without full retraining.
* **Practical scripting (internal)**: Lightweight shell scripts used to automate experimental workflows (not included in this repo). Feel free to reach out if you'd like access to these scripts or details on how we structured the workflows.

This fork is intended for researchers interested in exploring the impact of this approach at different scales of model, training length, data quality, and quantity. Results can be found here: (result comming soon)

---

## Requirements

Same as the original repo:

* PyTorch >= 2.1, Python 3.9+
* `pip install click Pillow psutil requests scipy tqdm diffusers==0.26.3 accelerate==0.27.2`

---

## Using Pre-trained Checkpoints

To run the second-stage training or evaluation, you can start from any pretrained checkpoint hosted by NVIDIA ([see ](https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/)[raw snapshots](https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/)).

Example to generate from the original model:

```bash
python generate_images.py --preset=edm2-img512-s-guid-fid --outdir=out
```

Or use `--net` to start from a specific intermediate checkpoint.

---

## Training on Curated Subsets

The dataset class has been extended to support training on filtered subsets:

```bash
python train_edm2.py \
    --data=datasets/img512-curated.zip \
    --preset=base-finetuning \
    --model2finetune=raw-snapshots/edm2-img512-s/network-snapshot-xxxxxxx.pkl \
    --freeze-down \
    --batch-gpu=32 \
    --duration=4Mi \
    --top-per-category=5
```

You can customize filtering via embedded scorers or external tools.

---

## Metric Evaluation

Supports multi-distribution evaluation with:

```bash
python calculate_metrics.py gen \
    --net=snapshot.pkl \
    --out=results \
    --out-images=image_save_path \
    --ref=https://.../imagenet.pkl \
    --ref=https://.../custom_dataset.pkl
```

---

## Citation

Please cite the original works:

```bibtex
@inproceedings{Karras2024edm2,
  title     = {Analyzing and Improving the Training Dynamics of Diffusion Models},
  author    = {Tero Karras and Miika Aittala and Jaakko Lehtinen and Janne Hellsten and Timo Aila and Samuli Laine},
  booktitle = {Proc. CVPR},
  year      = {2024},
}

@inproceedings{Karras2024autoguidance,
  title     = {Guiding a Diffusion Model with a Bad Version of Itself},
  author    = {Tero Karras and Miika Aittala and Tuomas Kynkäänniemi and Jaakko Lehtinen and Timo Aila and Samuli Laine},
  booktitle = {Proc. NeurIPS},
  year      = {2024},
}
```

---

## License

Same as the original [repo: ](http://creativecommons.org/licenses/by-nc-sa/4.0/)[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## Contact

For questions about this fork, feel free to reach out directly (not NVIDIA).
