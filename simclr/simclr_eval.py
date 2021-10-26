from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.nn.functional as F
import higher
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torch.autograd import grad
from tqdm import tqdm
from simclr_models import *
from simclr_datasets import *
from nt_xent import NTXentLoss

from torch.backends import cudnn
cudnn.deterministic = True
cudnn.benchmark = False

import argparse

parser = argparse.ArgumentParser(description='Eval SIMCLR ECG')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--ex', type=int, default=500, help='num data points')

parser.add_argument('--finetune_lr', type=float, default=1e-3)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--studentarch', type=str, default='resnet18')
parser.add_argument('--dataset', type=str, default='ecg')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--savefol', type=str, default='simclr-ecg-eval')
parser.add_argument('--transfer_eval', action='store_true')
parser.add_argument('--checkpoint', type=str)

args = parser.parse_args()

torch.manual_seed(args.runseed)
torch.multiprocessing.set_sharing_strategy('file_system')


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.transfer_eval:
    args.savefol += f'-transfereval-{args.ex}ex'
else:
    args.savefol += f'-lineval-{args.ex}ex'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def model_saver(epoch, student, head, teacher, pt_opt, pt_sched, ft_opt, hyp_opt, path):
    torch.save({
        'student_sd': student.state_dict(),
        'teacher_sd': teacher.state_dict() if teacher is not None else None,
        'head_sd': head.state_dict(),
        'ft_opt_state_dict': ft_opt.state_dict(),
    }, path + f'/checkpoint_epoch{epoch}.pt')


def get_save_path():
    modfol =  f"""seed{args.seed}-runseed{args.runseed}-student{args.studentarch}-ftlr{args.finetune_lr}-epochs{args.epochs}-ckpt{args.checkpoint}"""
    pth = os.path.join(args.savefol, modfol)
    os.makedirs(pth, exist_ok=True)
    return pth

def get_loss(student,head,teacher,  x, y):
    head_op = head(student.logits(x))
    l_obj = nn.BCEWithLogitsLoss()
    clf_loss = l_obj(head_op, y)
    y_loss_stud = clf_loss
    acc_stud = 0 #torch.mean(torch.sigmoid(head_op) > 0.5 * y).item()
    return y_loss_stud, acc_stud

# Utility function to update lossdict
def update_lossdict(lossdict, update, action='append'):
    for k in update.keys():
        if action == 'append':
            if k in lossdict:
                lossdict[k].append(update[k])
            else:
                lossdict[k] = [update[k]]
        elif action == 'sum':
            if k in lossdict:
                lossdict[k] += update[k]
            else:
                lossdict[k] = update[k]
        else:
            raise NotImplementedError
    return lossdict


from sklearn.metrics import roc_auc_score

# Evaluate student on complete train/test set.
def eval_student(student, head, dl):
    student.eval()
    net_loss = 0
    correct = 0
    y_pred = []
    y_true = []
    l_obj = nn.BCEWithLogitsLoss(reduction='sum')
#     clf_loss = l_obj(head_op, y)
    with torch.no_grad():
        for data, target in dl:
            y_true.append(target.detach().cpu().numpy())
            data, target = data.to(device), target.to(device)
            output = head(student.logits(data))
            net_loss += l_obj(output, target).item()  # sum up batch loss
            y_pred.append(output.detach().cpu().numpy())
#             pred = torch.sigmoid(output) > 0.5
#             correct += torch.sum(pred == target).item()

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    net_loss /= len(dl.dataset)
#     acc = 100. * correct / len(dl.dataset * y_pred.shape[1])
    
    roc_list = []
    for i in range(y_true.shape[1]):
        try:
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                roc_list.append(roc_auc_score(y_true[:,i], y_pred[:,i]))
            else:
                roc_list.append(np.nan)
        except ValueError:
            roc_list.append(np.nan)

    print('Average loss: {:.4f}, AUC: {:.4f}'.format(net_loss, np.mean(roc_list)))
    return {'epoch_loss': net_loss, 'auc' : roc_list}

import copy

def do_train_step(student, head, optimizer, x, y):
    student.eval()

    x = x.to(device)
    y = y.to(device)
    
    loss, acc = get_loss(student,  head, None, x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc

def train():
    ft_loss_meter = AverageMeter()
    ft_acc_meter = AverageMeter()
    
    DSHandle = ECGDataSetWrapper(args.batch_size)
    pretrain_dl, train_dl, val_dl, test_dl, _, NUM_TASKS_FT = DSHandle.get_data_loaders(args, evaluate=True)
    
    torch.manual_seed(args.runseed)
    import random
    random.seed(args.runseed)
    np.random.seed(args.runseed)

    if args.studentarch == 'resnet18':
        student = ecg_simclr_resnet18().to(device)
        head = MultitaskHead(256, NUM_TASKS_FT).to(device)
    else:
        raise NotImplementedError

    if args.checkpoint is None:
        print("No checkpoint! Training from scratch")
    else:
        ckpt = torch.load(args.checkpoint)
        student.load_state_dict(ckpt['student_sd'], strict=False)
#         head.load_state_dict(ckpt['head_sd']) # Not loading head, so commented out
        print("Loading student; not doing pretraining")

    if args.transfer_eval:
        finetune_optim = torch.optim.Adam(list(head.parameters()) + list(student.parameters()), lr=args.finetune_lr)
    else:
        finetune_optim = torch.optim.Adam(head.parameters(), lr=args.finetune_lr)

    stud_finetune_train_ld = {'loss' : [], 'acc' : []}
    stud_finetune_val_ld = {'loss' : [], 'acc' : []}
    stud_finetune_test_ld = {}

    for n in range(args.epochs):
        progress_bar = tqdm(train_dl)
        for i, (x,y) in enumerate(progress_bar):
            progress_bar.set_description('Finetune Epoch ' + str(n))
            ft_train_loss, ft_train_acc = do_train_step(student, head, finetune_optim, x, y)
            ft_loss_meter.update(ft_train_loss)
            ft_acc_meter.update(ft_train_acc)
            progress_bar.set_postfix(
                    finetune_train_loss='%.4f' % ft_loss_meter.avg ,
                    finetune_train_acc='%.4f' % ft_acc_meter.avg ,
                )
            # append to lossdict
            stud_finetune_train_ld['loss'].append(ft_train_loss)
            stud_finetune_train_ld['acc'].append(ft_train_acc)

            
        ft_test_ld = eval_student(student,head,  test_dl)
        stud_finetune_test_ld = update_lossdict(stud_finetune_test_ld, ft_test_ld)
        
        ft_val_ld = eval_student(student,head,  val_dl)
        stud_finetune_val_ld = update_lossdict(stud_finetune_val_ld, ft_val_ld)

        ft_train_ld = eval_student(student,head, train_dl)
        stud_finetune_train_ld = update_lossdict(stud_finetune_train_ld, ft_train_ld)
        # save the logs
        tosave = {
            'finetune_train_ld' : stud_finetune_train_ld,
            'finetune_val_ld' : stud_finetune_val_ld,
            'finetune_test_ld' : stud_finetune_test_ld,
        }
        torch.save(tosave, os.path.join(get_save_path(), 'eval_logs.ckpt'))
        # if n % 5 == 0:
        #     model_saver(n, student, head, pretrain_optim, finetune_optim, save_path)
        #     print(f"Saved model at epoch {n}")
        ft_loss_meter.reset()
        ft_acc_meter.reset()
    return student, head, finetune_optim


res = train()





