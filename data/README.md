# Dataset Preparation

Raw datasets and generated caches are not included in this repository.

## CIFAR-10 and CIFAR-100

The scripts use torchvision with automatic downloading enabled.
Use the repository data directory as the dataset root:

    --root data

Downloaded files remain under data/ and are ignored by Git.

## CINIC-10

Download and extract CINIC-10 manually. The directory must contain:

    data/CINIC-10/train/
    data/CINIC-10/valid/
    data/CINIC-10/test/

Run the corresponding script with:

    --root data --cinic10_path data/CINIC-10

Generated caches under data/cinic10_processed/ are ignored by Git.

## FEMNIST

Provide FEMNIST as a Hugging Face save_to_disk directory or a local
Parquet shard from the flwrlabs/femnist dataset artifact.

Supported locations include:

    data/femnist_hf/
    data/femnist_processed/
    data/hf_cache/

An explicit path can also be provided:

    --root data --femnist_path PATH_TO_FEMNIST

FEMNIST requires datasets, pyarrow, and Pillow.

## Data Policy

Users must download each dataset and comply with its original license.
This repository does not redistribute raw dataset files.
