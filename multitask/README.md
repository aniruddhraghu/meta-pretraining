# Meta-Parameterized Multitask PT
This folder contains code to run the meta-parameterized multitask PT experiments in the paper. Note that most of the dataloading, model definition, and training code is based on the implementation of `Strategies for Pre-training Graph Neural Nets', Hu et al., ICLR 2020 at [this link](https://github.com/snap-stanford/pretrain-gnns/).


## Getting started
Follow the installation instructions from Hu et al., ICLR 2020 at [this link](https://github.com/snap-stanford/pretrain-gnns/). Download the biological dataset, and all the python dependencies.  Also install the `higher` library: `pip install higher`. Once you have downloaded the dataset, make sure the `root_supervised` variable in the training scripts and the `util.py` script is set to be that path.


## Pre-training

### Full FT Access
To train the full FT access model, run:
```python pretrain_supervised_weighting.py --gpu 0 --savefol exw-adamhyper```

### Partial FT Access
To train the partial FT access model with 0.5 of the total data, run:
```python pretrain_supervised_weighting.py --gpu 0 --savefol exw-adamhyper --smallft 0.5```

To train the partial FT access model on a subset of the FT tasks, say fold 0 (tasks 0-29 in meta PT, tasks 30-39 at FT time), run:
```python pretrain_supervised_multitask5030.py --gpu 0 --fold 0```

## Fine-tuning

### Full FT Access and Partial FT access with small MetaFT dataset
To fine-tune the full FT access model or the model using a small MetaFT dataset, run the below command, replacing the `OUTPUT_FILENAME` by the desired output filename, `SEED` by the fine-tuning seed, `LR` by the FT learning rate, and `PATH/TO/CHECKPOINT` by the path to the pre-trained checkpoint. 

```python finetune.py --device 0 --filename OUTPUT_FILENAME  --runseed SEED --lr LR --model_file PATH/TO/CHECKPOINT```

By default, this does full transfer. If you want to do linear evaluation, add the `--lineval` flag to the end of the above command. We used an LR of 1e-5 for full transfer, and 1e-4 for linear eval.


### Partial FT Access model with subset of tasks
To fine-tune, run the below command, replacing the `OUTPUT_FILENAME` by the desired output filename, `SEED` by the fine-tuning seed, `LR` by the FT learning rate, `PATH/TO/CHECKPOINT` by the path to the pre-trained checkpoint, and `FOLD` by the desired fold.

```python finetune_ft30.py --device 0 --filename OUTPUT_FILENAME  --runseed SEED --lr LR --model_file PATH/TO/CHECKPOINT --fold FOLD```

By default, this does full transfer. If you want to do linear evaluation, add the `--lineval` flag to the end of the above command. We used an LR of 1e-5 for full transfer, and 1e-4 for linear eval.
