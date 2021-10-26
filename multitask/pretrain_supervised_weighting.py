import argparse

from splitters import random_split, species_split
from loader import BioDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from dataloader import DataLoaderFinetune
from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

import pandas as pd

from util import combine_dataset
from model import GNN, GNN_graphpred, GNNHead, TaskWeight

from PIL import Image
import numpy as np
import os
import torch.utils.data as utils
import higher
import pickle
from torch.autograd import grad
import random

parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=1)
parser.add_argument('--pretrain_steps', type=int, default = 10)
parser.add_argument('--finetune_steps', type=int, default = 1)
parser.add_argument('--neumann', type=int, default=1)

parser.add_argument('--pretrain_lr', type=float, default=1e-4,
                    help='learning rate (default: 0.001)')

parser.add_argument('--finetune_lr', type=float, default=1e-4)

parser.add_argument('--hyper_lr', type=float, default=1)


parser.add_argument('--decay', type=float, default=0,
                    help='weight decay (default: 0)')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5).')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='embedding dimensions (default: 300)')
parser.add_argument('--dropout_ratio', type=float, default=0.2,
                    help='dropout ratio (default: 0.2)')
parser.add_argument('--graph_pooling', type=str, default="mean",
                    help='graph level pooling (sum, mean, max, set2set, attention)')
parser.add_argument('--JK', type=str, default="last",
                    help='how the node features across layers are combined. last, sum, max or concat')
parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
parser.add_argument('--savefol', type=str, default='exw-adamhyper')
parser.add_argument('--gnn_type', type=str, default="gin")
parser.add_argument('--num_workers', type=int, default = 0, help='number of workers for dataset loading')
parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
parser.add_argument('--split', type=str, default = "species", help='Random or species split')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--smallft', type=float, default = 1, help='split the val set down or not')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--ft_bn', action='store_true')
args = parser.parse_args()


torch.manual_seed(0)
np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

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


# Need to set this!
root_supervised = '/path/to/dataset/'

dataset = BioDataset(root_supervised, data_type='supervised')


print("Making PT dataset...")
if args.split == "random":
    print("random splitting")
    train_dataset, valid_dataset, test_dataset = random_split(dataset, seed = args.seed)
    print(train_dataset)
    print(valid_dataset)
    pretrain_dataset = combine_dataset(train_dataset, valid_dataset)
    print(pretrain_dataset)
elif args.split == "species":
    print("species splitting")
    trainval_dataset, test_dataset = species_split(dataset)
    test_dataset_broad, test_dataset_none, _ = random_split(test_dataset, seed = args.seed, frac_train=0.5, frac_valid=0.5, frac_test=0)
    print(trainval_dataset)
    print(test_dataset_broad)
    pretrain_dataset = combine_dataset(trainval_dataset, test_dataset_broad)            
    print(pretrain_dataset)
else:
    raise ValueError("Unknown split name.")

pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
pretrain_val_loader = None
NUM_TASKS_PT = len(pretrain_dataset[0].go_target_pretrain)


print("Making FT dataset...")
if args.split == "random":
    print("random splitting")
    train_dataset, valid_dataset, test_dataset = random_split(dataset, seed = args.seed) 
elif args.split == "species":
    trainval_dataset, test_dataset = species_split(dataset)
    train_dataset, valid_dataset, _ = random_split(trainval_dataset, seed = args.seed, frac_train=0.85, frac_valid=0.15, frac_test=0)
    test_dataset_broad, test_dataset_none, _ = random_split(test_dataset, seed = args.seed, frac_train=0.5, frac_valid=0.5, frac_test=0)
    print("species splitting")
else:
    raise ValueError("Unknown split name.")
    
if args.smallft != 1:
    random.seed(0)
    torch.manual_seed(0)
    len_train = int(args.smallft*len(train_dataset))
    len_val = int(args.smallft*len(valid_dataset))
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [len_train, len(train_dataset)-len_train])
    valid_dataset, _ = torch.utils.data.random_split(valid_dataset, [len_val, len(valid_dataset)-len_val])
    

finetune_train_loader = DataLoaderFinetune(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
finetune_val_loader = DataLoaderFinetune(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

if args.split == "random":
    test_loader = DataLoaderFinetune(test_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
else:
    ### for species splitting
    finetune_test_loader_easy = DataLoaderFinetune(test_dataset_broad, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
    finetune_test_loader_hard = DataLoaderFinetune(test_dataset_none, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)

NUM_TASKS_FT = len(dataset[0].go_target_downstream)



def model_saver(epoch, student, head, teacher, pt_opt, ft_opt, hyp_opt, path):
    torch.save({
        'student_sd': student.state_dict(),
        'teacher_sd': teacher.state_dict(),
        'head_sd': head.state_dict(),
        'pt_opt_state_dict': pt_opt.state_dict(),
        'ft_opt_state_dict': ft_opt.state_dict(),
        'hyp_opt_state_dict': hyp_opt.state_dict(),
    }, path + f'/checkpoint_epoch{epoch}.pt')


def get_save_path():
    modfol =  f"""ptlr{args.pretrain_lr}-ftlr{args.finetune_lr}-hyplr{args.hyper_lr}-warmup{args.warmup_epochs}-pt_steps{args.pretrain_steps}-ft_steps{args.finetune_steps}-neumann{args.neumann}-ft_bn{args.ft_bn}-smallft{args.smallft}"""
    pth = os.path.join(args.savefol, modfol)
    os.makedirs(pth, exist_ok=True)
    return pth

def zero_hypergrad(hyper_params):
    """

    :param get_hyper_train:
    :return:
    """
    current_index = 0
    for p in hyper_params:
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0
        current_index += p_num_params


def store_hypergrad(hyper_params, total_d_val_loss_d_lambda):
    """

    :param get_hyper_train:
    :param total_d_val_loss_d_lambda:
    :return:
    """
    current_index = 0
    for p in hyper_params:
        p_num_params = np.prod(p.shape)
        p.grad = total_d_val_loss_d_lambda[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params


def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model, head):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner

    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter

        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, list(model.parameters())+list(head.parameters()), grad_outputs=counter.view(-1), retain_graph=True))
        # hessian_term[hessian_term == None] = 0
        counter = old_counter - elementary_lr * hessian_term

        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner

def get_hyper_train_flat(hyper_params):
    return torch.cat([p.view(-1) for p in hyper_params])

def gather_flat_grad(loss_grad):
    return torch.cat([p.reshape(-1) for p in loss_grad]) #g_vector


# Forward pass on student and teacher, getting neg log lik of label and batch acc
def get_loss(student,head,teacher,  batch, ft):
    batch = batch.to(device)
    feats = student.logits(batch)
    pi_stud = student.head(feats)
    head_op = head(feats)
    if ft:
        y = batch.go_target_downstream.view(head_op.shape).to(torch.float64)
        l_obj = nn.BCEWithLogitsLoss()
        loss = l_obj(head_op.double(), y)
        y_loss_stud = loss + 0*torch.sum(pi_stud.double())
    else:
        y = batch.go_target_pretrain.view(pi_stud.shape).to(torch.float64)
        comm = teacher.forward()
        l_obj = nn.BCEWithLogitsLoss(reduction = 'none')
        loss = l_obj(pi_stud.double(), y)
        # now apply the classweight
        comm = comm.unsqueeze(0)
        y_loss_stud = torch.mean(comm*loss) + 0*torch.sum(head_op) * 0*torch.sum(loss) # force dependence on other things so autograd does not complain
    return y_loss_stud, 0


def hyper_step(model, head, teacher, hyper_params, pretrain_loader, optimizer, d_val_loss_d_theta, elementary_lr, neum_steps):
    zero_hypergrad(hyper_params)
    num_weights = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    num_hypers = sum(p.numel() for p in hyper_params)

    d_train_loss_d_w = torch.zeros(num_weights).cuda()
    model.train(), model.zero_grad(),head.train(), head.zero_grad()

    # NOTE: This should be the pretrain set: gradient of PRETRAINING loss wrt pretrain parameters.
    for batch_idx, batch in enumerate(pretrain_loader):
        batch = batch.to(device)
        train_loss, _ = get_loss(model, head, teacher, batch, ft=False)
        optimizer.zero_grad()
        d_train_loss_d_w += gather_flat_grad(grad(train_loss, list(model.parameters())+list(head.parameters()), 
                                                  create_graph=True))
        break
    optimizer.zero_grad()

    # Initialize the preconditioner and counter
    preconditioner = d_val_loss_d_theta

    preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr,
                                                          neum_steps, model, head)
    
    # THIS SHOULD BE PRETRAIN LOSS AGAIN.
    indirect_grad = gather_flat_grad(
        grad(d_train_loss_d_w, hyper_params, grad_outputs=preconditioner.view(-1)))
    hypergrad = indirect_grad # no direct grad term

    zero_hypergrad(hyper_params)
    store_hypergrad(hyper_params, -hypergrad)
    return hypergrad

def do_pretrain(student, head, teacher, optimizer, batch):
    student.train()

    batch = batch.to(device)

    loss, acc = get_loss(student,  head, teacher, batch, ft=False)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc



def inner_loop_finetune(student, head, teacher, optimizer, train_dl, val_dl, num_steps):
    stud_loss = 0.
    stud_acc = 0.

    if args.ft_bn:
        student.train()
    else:
        student.eval() # BN should be eval model when we do only head unrolling!
    teacher.train()
    
    for i, batch in enumerate(train_dl):
        batch = batch.to(device)
        y_loss, acc = get_loss(student, head, teacher, batch, ft=True)
        optimizer.step(y_loss)
        # logging
        stud_loss += y_loss.item()
        stud_acc += acc

        if i == num_steps - 1:
            break
    stud_loss /= num_steps
    stud_acc /= num_steps
        
    # Now compute the val loss
    avgloss = None
    avgacc = None
    for i, batch in enumerate(val_dl):
        batch = batch.to(device)
        y_loss, acc = get_loss(student, head,  teacher, batch, ft=True)

        if avgloss is None:
            avgloss = y_loss
            avgacc = acc
        else:
            avgloss += y_loss
            avgacc += acc
        break
    # print(avgloss.item())
    # Now compute a finetuning gradient
    ft_grad = torch.autograd.grad(avgloss, list(student.parameters())+list(head.parameters(time=0)), allow_unused=True)
    return (stud_loss, stud_acc), (avgloss, avgacc), ft_grad, head



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



# Evaluate student on complete train/test set.
def eval_student(student, head, dl):
    student.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(dl, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = head(student.logits(batch))

        y_true.append(batch.go_target_downstream.view(pred.shape).detach().cpu())
        y_scores.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_scores = torch.cat(y_scores, dim = 0).numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        try:
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                roc_list.append(roc_auc_score(y_true[:,i], y_scores[:,i]))
            else:
                roc_list.append(np.nan)
        except ValueError:
            roc_list.append(np.nan)

    return {'auc':np.array(roc_list)} #y_true.shape[1]


import copy
import time
def do_ft_head(student, head, optimizer, dl):
    if args.ft_bn:
        student.train()
    else:
        student.eval()
    for batch in dl:
        batch = batch.to(device)
        feats = student.logits(batch).detach()
        head_op = head(feats)

        l_obj = nn.BCEWithLogitsLoss()
        y = batch.go_target_downstream.view(head_op.shape).to(torch.float64)
        loss = l_obj(head_op.double(), y)
        acc = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        break
    return loss.item(), acc



def train(pretrain_dl, train_dl, val_dl, test_dl_easy, test_dl_hard):
    # also creates save path
    save_path = get_save_path()
    
    student = GNN_graphpred(args.num_layer, args.emb_dim, NUM_TASKS_PT, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    student = student.to(device)
    head = GNNHead(2*args.emb_dim, NUM_TASKS_FT).to(device)
    
    teacher = TaskWeight(num_classes=NUM_TASKS_PT).to(device)

    hyp_params = list(teacher.parameters())

    pretrain_optim = torch.optim.Adam(student.parameters(), lr=args.pretrain_lr, weight_decay=args.decay)

    finetune_optim = torch.optim.Adam(head.parameters(), lr=args.finetune_lr)
    hyp_optim = torch.optim.Adam(hyp_params, lr=args.hyper_lr)

    stud_pretrain_ld = {'loss' : [], 'acc' : [] }
    stud_finetune_train_ld = {'loss' : [], 'acc' : []}
    stud_finetune_val_ld = {'loss' : [], 'acc' : []}
    stud_finetune_test_ld_easy = {}
    stud_finetune_test_ld_hard = {}

    num_finetune_steps = args.finetune_steps
    num_neumann_steps = args.neumann

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        student.load_state_dict(ckpt['student_sd'])
        teacher.load_state_dict(ckpt['teacher_sd'])
        head.load_state_dict(ckpt['head_sd'])
        pretrain_optim.load_state_dict(ckpt['pt_opt_state_dict'])
        finetune_optim.load_state_dict(ckpt['ft_opt_state_dict'])
        hyp_optim.load_state_dict(ckpt['hyp_opt_state_dict'])
        load_ep = int(os.path.split(args.checkpoint)[-1][16:-3]) + 1
        print(f"Loaded checkpoint {args.checkpoint}, epoch {load_ep}")
    else:
        load_ep = 0
    steps = 0
    for n in range(load_ep, args.epochs):
        pt_loss_meter = AverageMeter()
        pt_acc_meter = AverageMeter()
        ft_train_loss_meter = AverageMeter()
        ft_train_acc_meter = AverageMeter()
        ft_val_loss_meter = AverageMeter()
        ft_val_acc_meter = AverageMeter()

        progress_bar = tqdm(pretrain_dl)
        for i, batch in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(n))
            zero_hypergrad(hyp_params)

            if n < args.warmup_epochs:
                student.zero_grad(), head.zero_grad()
                pt_loss, pt_acc = do_pretrain(student, head, teacher, pretrain_optim, batch)
                pt_loss_meter.update(pt_loss)
                pt_acc_meter.update(pt_acc)

                student.zero_grad(), head.zero_grad()
                ft_train_loss, ft_train_acc = do_ft_head(student, head, finetune_optim, train_dl)
                ft_train_loss_meter.update(ft_train_loss)
                ft_train_acc_meter.update(ft_train_acc)
                ft_val_loss, ft_val_acc, hypg = 0,0,0
            else:
                pt_loss, pt_acc = do_pretrain(student, head, teacher, pretrain_optim, batch)
                pt_loss_meter.update(pt_loss)
                pt_acc_meter.update(pt_acc)
                if steps % args.pretrain_steps == 0:
                    with higher.innerloop_ctx(head, finetune_optim, copy_initial_weights=True) as (fnet, diffopt):
                        (ft_train_loss, ft_train_acc), (ft_val_loss, ft_val_acc), ft_grad, fnet = \
                                            inner_loop_finetune(student, fnet, teacher, diffopt, train_dl, val_dl, num_finetune_steps)
                        head.load_state_dict(fnet.state_dict())

                    ft_grad = gather_flat_grad(ft_grad)
                    for param_group in pretrain_optim.param_groups:
                        cur_lr = param_group['lr']
                        break

                    hypg = hyper_step(student, head, teacher, hyp_params, pretrain_loader, pretrain_optim, ft_grad, cur_lr, num_neumann_steps)
                    hypg = hypg.norm().item()

                    hyp_optim.step()
                    ft_train_loss_meter.update(ft_train_loss)
                    ft_train_acc_meter.update(ft_train_acc)
                    ft_val_loss_meter.update(ft_val_loss)
                    ft_val_acc_meter.update(ft_val_acc)
                else:
                    ft_train_loss, ft_train_acc = do_ft_head(student, head, finetune_optim, train_dl)
                    ft_train_loss_meter.update(ft_train_loss)
                    ft_train_acc_meter.update(ft_train_acc)
                    ft_val_loss, ft_val_acc, hypg = 0,0,0


            steps += 1
            progress_bar.set_postfix(
                    pt_l='%.4f' % pt_loss_meter.avg ,
                    pt_a='%.4f' % pt_acc_meter.avg ,
                    ft_tl='%.4f' % ft_train_loss_meter.avg ,
                    ft_ta='%.4f' % ft_train_acc_meter.avg ,
                    ft_vl='%.4f' % ft_val_loss_meter.avg ,
                    ft_va='%.4f' % ft_val_acc_meter.avg ,
                    hyp_norm='%.6f' % hypg
                )

            # append to lossdict
            stud_pretrain_ld['loss'].append(pt_loss)
            stud_pretrain_ld['acc'].append(pt_acc)
            stud_finetune_train_ld['loss'].append(ft_train_loss)
            stud_finetune_train_ld['acc'].append(ft_train_acc)
            stud_finetune_val_ld['loss'].append(ft_val_loss)
            stud_finetune_val_ld['acc'].append(ft_val_acc)

            
        ft_test_ld = eval_student(student,head,  test_dl_easy)
        print("Easy test")
        print(ft_test_ld)
        stud_finetune_test_ld_easy = update_lossdict(stud_finetune_test_ld_easy, ft_test_ld)

        ft_test_ld = eval_student(student,head,  test_dl_hard)
        print("Hard test")
        print(ft_test_ld)
        stud_finetune_test_ld_hard = update_lossdict(stud_finetune_test_ld_hard, ft_test_ld)

        ft_val_ld = eval_student(student, head, val_dl)
        print("Val")
        print(ft_val_ld)
        stud_finetune_val_ld = update_lossdict(stud_finetune_val_ld, ft_val_ld)


        if pretrain_val_loader is not None:
            print("Evaluating on pretrain val set")
            eval_student(student, None, pretrain_val_loader)

        tosave = {
            'pretrain_ld' : stud_pretrain_ld,
            'finetune_train_ld' : stud_finetune_train_ld,
            'finetune_val_ld' : stud_finetune_val_ld,
            'finetune_test_ld_easy' : stud_finetune_test_ld_easy,
            'finetune_test_ld_hard' : stud_finetune_test_ld_hard,
        }
        torch.save(tosave, os.path.join(save_path, 'logs.ckpt'))
        if n % 20 == 0 or n == args.epochs - 1:
            model_saver(n, student, head, teacher, pretrain_optim, finetune_optim, hyp_optim, save_path)
            print(f"Saved model at epoch {n}")
    return student, head, teacher, pretrain_optim, finetune_optim, hyp_optim


res = train(pretrain_loader, finetune_train_loader, finetune_val_loader, finetune_test_loader_easy, finetune_test_loader_hard)

if args.save:
    save_path = get_save_path()
    student, head, teacher, pretrain_optim, finetune_optim, hyp_optim = res
    model_saver(args.epochs, student, head, teacher, pretrain_optim, finetune_optim, hyp_optim, save_path)







