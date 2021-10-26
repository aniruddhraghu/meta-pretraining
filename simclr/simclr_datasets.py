import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import random
# np.random.seed(0)

import pandas as pd
import wfdb
import ast

    
class ECGSimCLR(Dataset):
    def __init__(self, x, y, transform, simclr=True):
        super(ECGSimCLR,self).__init__()
        # do some padding here.
        if x.shape[1] != 1024 and x.shape[1] == 1000:
            # pad
            x = np.pad(x, [[0,0], [0,24], [0,0]])
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.transform = transform
        self.simclr = simclr

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.simclr:
            return x
        else:
            sample = (x, y)
            return sample


class ECGDataSetWrapper(object):

    def __init__(self, batch_size, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loaders(self, args, evaluate = False):
        
        def load_raw_data(df, sampling_rate, path):
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
            data = np.array([signal for signal, meta in data])
            return data

        def aggregate_diagnostic(y_dic):
            tmp = np.zeros(5)
            idxd = {'NORM' : 0, 'MI' : 1, 'STTC' : 2, 'CD' : 3, 'HYP' : 4}
            for key in y_dic.keys():
                if key in agg_df.index:
                    cls = agg_df.loc[key].diagnostic_class
                    tmp[idxd[cls]] = 1
            return tmp

        path = 'path/to/dataset/'
        sampling_rate=100

        # load and convert annotation data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data(Y, sampling_rate, path)

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        
        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        # Split data into train and test
        test_fold = 10
        # Train
        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        y_train = np.stack(y_train, axis=0)
        
        # Test
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
        y_test = np.stack(y_test, axis=0)

        data_augment = self.get_simclr_pipeline_transform()
        data_augment = SimCLRDataTransform(data_augment)
        
        FT_TASKS = 5
        
        # Normalisation: follow PTB-XL demo code. Do zero mean, unit var normalisation across all leads, timesteps, and patients
        meansig = np.mean(X_train.reshape(-1))
        stdsig = np.std(X_train.reshape(-1))
        X_train = (X_train - meansig)/stdsig
        X_test = (X_test - meansig)/stdsig

        pretrain_dataset = ECGSimCLR(X_train, y_train, data_augment)

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        rng = np.random.RandomState(args.seed)
        idxs = np.arange(len(y_train))
        rng.shuffle(idxs)
        
        
        if args.ex >= 50:
            train_samp = int(0.8*args.ex)
            val_samp = args.ex - train_samp
        else:
            if args.ex == 25:
                train_samp = 15
                val_samp = 10
            elif args.ex == 10:
                train_samp = val_samp = 5

        train_idxs = idxs[:train_samp]
        val_idxs = idxs[train_samp:train_samp+val_samp]
        
        ft_train = ECGSimCLR(X_train[train_idxs], y_train[train_idxs], transform=None, simclr=False)
        ft_val = ECGSimCLR(X_train[val_idxs], y_train[val_idxs], transform=None, simclr=False)
        ft_test = ECGSimCLR(X_test, y_test, transform=None, simclr=False)

        pretrain_loader = torch.utils.data.DataLoader(dataset=pretrain_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0, 
                                               drop_last=True)
        ft_train_loader = torch.utils.data.DataLoader(dataset=ft_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)
        ft_val_loader = torch.utils.data.DataLoader(dataset=ft_val,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)
        ft_test_loader = torch.utils.data.DataLoader(dataset=ft_test,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)                                            

        return pretrain_loader, ft_train_loader, ft_val_loader, ft_test_loader, None, FT_TASKS

    def get_simclr_pipeline_transform(self):
        def rand_crop_ecg(ecg):
            cropped_ecg = ecg.copy()
            for j in range(ecg.shape[1]):
                crop_len = np.random.randint(len(ecg)) // 2
                crop_start = max(0, np.random.randint(-crop_len, len(ecg)))
                cropped_ecg[crop_start: crop_start + crop_len, j] = 0
            return cropped_ecg
        def rand_add_noise(ecg):
            noise_frac = np.random.rand() * .1
            return ecg + noise_frac * ecg.std(axis=0) * np.random.randn(*ecg.shape)
        data_transforms = [rand_crop_ecg, rand_add_noise]
        return data_transforms



class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = sample.copy()
        for t in self.transform:
            xi = t(xi)

        xj = sample.copy()
        for t in self.transform:
            xj = t(xj)

        return xi.astype(np.float32), xj.astype(np.float32)
