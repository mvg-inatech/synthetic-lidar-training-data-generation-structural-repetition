# Evaluation of the realism of the generated synthetic point cloud data

To evaluate the generated synthetic point cloud data, we use 3D semantic segmentation as a downstream deep learning task.

We use the following deep learning models:
* Point Transformer v3
* OA-CNN
* SparseUNet

Implementations are based on the pointcept codebase from this [Git commit](https://github.com/Pointcept/Pointcept/commit/a75c3e06aeadf200f6947358ed85211c47acfc3e).


## Environment setup

We use a [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) inside Visual Studio Code.

1. Open the project folder in the devcontainer.
1. A terminal opens automatically and you will see the progress of the dependency installation. See `postStartCommand` in the `devcontainer.json` file for details.
1. Close all open terminal windows and open a new one, s.t. PATH environment variable changes are applied.
1. `source .venv/bin/activate`

In case you get pointops import errors later, run:
1. `uv pip install libs/pointops/ --no-build-isolation`
(However, pointcept is part of the uv.lock file and should thus be installed automatically)


## Updating dependencies

We use Astral's [uv](https://docs.astral.sh/uv) for Python environment and dependency management.

In case you would like to update the dependencies, do the following:
1. Check `pyproject.toml` and adjust
1. Generate a new lockfile (`uv.lock`): `$ rm -f uv.lock && uv sync && uv lock`


## Preprocess our synthetic subway dataset
1. `source .venv/bin/activate`
1. `cd pointcept`
1. `python pointcept/datasets/preprocessing/subway/preprocess_synth_subway.py --dataset_root ../../part1_data_generation/sliced_tunnels/ --output_root data/subway --num_workers 4`

## Preprocess our real subway dataset
1. Copy the annotated real-world dataset to `data/unprocessed_real_data/`
1. `source .venv/bin/activate`
1. `cd pointcept`
1. `python pointcept/datasets/preprocessing/subway/preprocess_real_subway.py --dataset_root data/unprocessed_real_data/ --output_root data/subway --num_workers 4 --train_ratio 1 --val_ratio 0 --test_ratio 0`
1. The preprocessed files are now all stored in `data/subway/real_train`. You must now create the train/test split manually by moving selected files to `data/subway/real_test`. Do this manually to ensure equal balancing of tunnel cross-section types, LiDAR sensors, etc., given the small number of real scenes.
1. Edit `data/subway/real_dataset_info.json` accordingly

## Preprocess the STSD dataset
1. Download the STSD v1.1 dataset ([paper](https://doi.org/10.1016/j.tust.2024.105829)) via the download form in the paper's [GitHub repository](https://github.com/lichking2017/STSD) and save it to `pointcept/data/STSD v 1.1 unprocessed`
1. `source .venv/bin/activate`
1. `cd pointcept`
1. `python pointcept/datasets/preprocessing/stsd/preprocess_stsd.py --dataset_root "data/STSD v 1.1 unprocessed/las" --output_root data/stsd --num_workers 4 --train_ratio 0.8 --val_ratio 0 --test_ratio 0.2`

Please find our STSD splits here: [pointcept/data/stsd/dataset_info.json](pointcept/data/stsd/dataset_info.json).

## Running experiments
Certian experiments initialize Point Transformer v3 from large-scale pretrained weights on Sonata.
First, download the Sonata checkpoint from HuggingFace ([link](https://huggingface.co/facebook/sonata/blob/main/pretrain-sonata-v1m1-0-base.pth)). Save it to `exp/sonata/pretrain-sonata-v1m1-0-base.pth`.

We provide bash scripts to launch our experiments. Results will be stored in the `exp/` directory.
Script file names ending with `_test.sh` run the final evaluation. Script file names not ending with `_test.sh` run the training.
For each experiment, we provide separate scripts for the different model architectures.

1. `source .venv/bin/activate`
1. `cd pointcept`
1. `./scripts/exp<ExperimentToLaunch>.sh`

* Experiment 1: (obsolete, not reported in the paper)
* Experiment 2: Pretraining effect of PTv3.
* Experiment 3: Learning rate tuning.
* Experiment 4: Long training run of best found hyperparameters.
* Experiment 5: Training on real data only.
* Experiment 6: Merged training on synthetic+real data.
* Experiment 7: First pretrain on synthetic data, then finetune on real data.
* Experiment 8: Synthetic training data volume ablation.
* Experiment 9: Influence of replicating the LiDAR intensity value across three RGB channels.
* Experiment 10: Intensity input feature ablation.
* Experiment 11: First pretrain on synthetic data, then finetune on STSD.
* Experiment 12: Training on STSD only.
