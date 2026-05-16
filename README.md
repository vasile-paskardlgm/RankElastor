# PyTorch Implementation of "Expand More, Shrink Less: Shaping Effective-Rank Dynamics for Dense Scaling in Recommendation"

This repository contains the PyTorch implementation accompanying our **RankElastor**. The codebase is built on the [FuxiCTR](https://pypi.org/project/fuxictr/) library, which we gratefully acknowledge. Additional extensions can be implemented by following the guidelines provided in FuxiCTR.

## Environment

The code was tested in the following environment:

- Ubuntu 22.04  
- CUDA 11.8  
- Python 3.8  
- torch == 1.13.0  
- torchvision  
- torchaudio  
- fuxictr == 2.3.1
  - keras_preprocessing  
  - pandas  
  - PyYAML  
  - scikit-learn  
  - numpy  
  - h5py  
  - tqdm  
  - pyarrow  
  - **polars <= 1.0.0**

## Dataset Preparation

Since the raw **Avazu** and **Criteo** datasets are large (over 5 GB), we download them automatically when running the preparation script:

Since the raw `Avazu` and `Criteo` datasets are very large (over 5 GB), we opt to download them when running the code. Use the following command to obtain the data. You can also modify the storage location if desired.

```bash
bash 1.prepare.sh
```

If this script is not compatible with your environment, you can download the datasets manually:  
- [Download Criteo](https://huggingface.co/datasets/reczoo/Criteo_x1)
- [Download Avazu](https://huggingface.co/datasets/reczoo/Avazu_x4)

## Faster Training with Preprocessed Data

After the first run, FuxiCTR generates the dataset in `parquet` format, which can be found at `data/Avazu/avazu_x4_3bbbc4c9` (taking `Avazu` dataset as an example, you can replace it with your preferred path). To enable faster training, you should update the following entries in the dataset configuration files. For example, in `./config/dataset_config.yaml`, make these changes:

```yaml
avazu_x4_3bbbc4c9:
    data_format: parquet # original: csv
    ...
    ...
    rebuild_dataset: false # original: true
    test_data: ../../data/Avazu/avazu_x4_3bbbc4c9/test.parquet # original: test.csv
    train_data: ../../data/Avazu/avazu_x4_3bbbc4c9/train.parquet # original: train.csv
    valid_data: ../../data/Avazu/avazu_x4_3bbbc4c9/valid.parquet # original: valid.csv
```

## Code Structure

```
src/
├── utils.py
├── Emb_Tokenizer.py
├── Token_Mixing.py
├── Token_FFN.py
├── UniMixer_layer.py
├── UniMixer.py
├── Permutation.py
run_expid.py
shape_analyser.py
```

## Module Description

- **`src/utils.py`**  
  Utility functions, including the GLU-improved P-FFNs and other reusable components.
- **`src/Emb_Tokenizer.py`**  
  Tokenization modules used in RankMixer and our RankElastor, supporting multiple tokenization schemes.
- **`src/Token_Mixing.py`**  
  Token-mixing implementations shared by RankMixer and RankElastor, including extensible mixing variants.
- **`src/Token_FFN.py`**  
  Per-token FFN (P-FFN) implementations with multiple activation functions and computation variants.
- **`src/UniMixer_layer.py`**  
  Unified block architecture for token-transformation-based recommenders.
- **`src/UniMixer.py`**  
  FuxiCTR-style executable model class for the unified architecture (RankMixer + RankElastor).
- **`src/Permutation.py`**  
  Not used in this paper.
- **`run_expid.py`**  
  Experiment execution entry point, consistent with FuxiCTR workflows.
- **`shape_analyser.py`**  
  Utilities for analyzing statistical properties of internal representations.

## Model Training and Testing

Run experiments with:
```python
python run_expid.py --expid ${experiment_id} --gpu 1
```
The hyperparameters for each experiment can be found in `model_config.yaml`. 

> **Note**: Since file paths may vary across computational platforms, please ensure that you set the correct paths in the code according to your environment.

## Running Custom Experiments

The model is configured through `model_config.yaml`, which specifies key hyperparameters include:
- number of tokens
- token dimension
- number of stacked blocks
- mixing configuration
- P-FFN configuration
- ...

To run custom experiments, define a new `experiment_id`.


## Acknowledgment

We gratefully acknowledge the open-source contributions of:
- [FuxiCTR](https://github.com/reczoo/FuxiCTR)
- [BARS](https://github.com/reczoo/BARS)
- [Multi-Embedding](https://github.com/thuml/Multi-Embedding)
- [AutoInt](https://github.com/shichence/AutoInt)
- [torchrec](https://github.com/meta-pytorch/torchrec)
- [recommenders](https://github.com/recommenders-team/recommenders)
- [DeepCTR](https://github.com/shenweichen/DeepCTR)