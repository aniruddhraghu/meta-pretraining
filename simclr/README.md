# Meta-Parameterized SimCLR
This folder contains code to run the meta-parameterized SimCLR experiments in the paper.


## Getting started
Download the dataset following the instructions [here](https://www.physionet.org/content/ptb-xl/1.0.1/). Once you have downloaded the dataset, make sure the `path` variable in `simclr_datasets.py` to be that path to the data.

Install the core libraries: `pip install torch higher wfdb pandas numpy`


## Pre-training

**No augmentation learning:** To train a SimCLR model with default augmentations run:
```python simclr_ecg.py --warmup_epochs 100 --epochs 50 --teacherarch warpexmag --gpu GPU --seed SEED```

The warmup epochs being greater than the number of epochs means the augmentations are not optimized.


**Augmentation learning:** To train a SimCLR model and optimize augmentations, with `N` MetaFT examples, run:
```python simclr_ecg.py --warmup_epochs 1 --epochs 50 --teacherarch warpexmag --gpu GPU --seed SEED --ex N```



## Fine-tuning

To fine-tune a pre-trained model on `NFT` fine-tuning examples (FT dataset has `N` data points), with FT seed RUNSEED and dataset seed (i.e., PT seed) SEED:

`python simclr_eval.py --gpu GPU --checkpoint /PATH/TO/CHECKPOINT --transfer_eval --runseed RUNSEED --seed SEED --ex NFT`

Note: The partial FT access setting is when `NFT` is more than `N` from the PT.
