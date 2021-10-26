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


import argparse

parser = argparse.ArgumentParser(description='ECG SIMCLR IFT')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int)

parser.add_argument('--pretrain_lr', type=float, default=1e-4)
parser.add_argument('--finetune_lr', type=float, default=1e-4)
parser.add_argument('--hyper_lr', type=float, default=1e-4)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--ex', default=500, type=int)
parser.add_argument('--warmup_epochs', type=int, default=1)
parser.add_argument('--pretrain_steps', type=int, default = 10)
parser.add_argument('--finetune_steps', type=int, default = 1)
parser.add_argument('--studentarch', type=str, default='resnet18')
parser.add_argument('--teacherarch')
parser.add_argument('--dataset', type=str, default='ecg')
parser.add_argument('--neumann', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--savefol', type=str, default='simclr')
parser.add_argument('--save', action='store_false')
parser.add_argument('--no_probs', action='store_true')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--teach_checkpoint', type=str)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.multiprocessing.set_sharing_strategy('file_system')

if args.no_probs:
    args.savefol += 'determ'

args.savefol += f'-{args.ex}ex'

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        'pt_opt_state_dict': pt_opt.state_dict(),
        'pt_sched_state_dict': pt_sched.state_dict(),
        'ft_opt_state_dict': ft_opt.state_dict(),
        'hyp_opt_state_dict': hyp_opt.state_dict() if teacher is not None else None,
    }, path + f'/checkpoint_epoch{epoch}.pt')


def get_save_path():
    modfol =  f"""seed{args.seed}-dataset{args.dataset}-student{args.studentarch}-teacher{args.teacherarch}-ptlr{args.pretrain_lr}-ftlr{args.finetune_lr}-hyplr{args.hyper_lr}-warmup{args.warmup_epochs}-pt_steps{args.pretrain_steps}-ft_steps{args.finetune_steps}-neumann{args.neumann}"""
    if args.teach_checkpoint:
        args.savefol += '-teachckpt'
        modfol = os.path.join(modfol, args.teach_checkpoint)
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
        counter = old_counter - elementary_lr * hessian_term

        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner

def get_hyper_train_flat(hyper_params):
    return torch.cat([p.view(-1) for p in hyper_params])

def gather_flat_grad(loss_grad):
    return torch.cat([p.reshape(-1) for p in loss_grad]) #g_vector

def get_loss(student,head,teacher,  x, y):
    head_op = head(student.logits(x))
    pi_stud = student(x)
    l_obj = nn.BCEWithLogitsLoss()
    clf_loss = l_obj(head_op, y)
    y_loss_stud = clf_loss + 0*torch.sum(pi_stud[0])+ 0*torch.sum(pi_stud[1])
    acc_stud = 0 #torch.mean(torch.sigmoid(head_op) > 0.5 * y).item()
    return y_loss_stud, acc_stud


def hyper_step(model, head, teacher, hyper_params, pretrain_loader, optimizer, d_val_loss_d_theta, elementary_lr, neum_steps):
    zero_hypergrad(hyper_params)
    num_weights = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())

    d_train_loss_d_w = torch.zeros(num_weights).to(device)
    model.train(), model.zero_grad(),head.train(), head.zero_grad()

    # NOTE: This should be the pretrain set: gradient of PRETRAINING loss wrt pretrain parameters.
    for batch_idx, (xis, xjs) in enumerate(pretrain_loader):
        xis = xis.to(device)
        xjs = xjs.to(device)
        if teacher is not None:
            xis = teacher(xis)
            xjs = teacher(xjs)
        train_loss= get_loss_simclr(model, xis, xjs)
        train_loss = train_loss + train_loss*head(model.logits(xis)).sum()*0
        optimizer.zero_grad()
        d_train_loss_d_w += gather_flat_grad(grad(train_loss, list(model.parameters())+list(head.parameters()), 
                                                  create_graph=True, allow_unused=True))
        break
    optimizer.zero_grad()

    # Initialize the preconditioner and counter
    preconditioner = d_val_loss_d_theta

    preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr,
                                                          neum_steps, model, head)

    
    # THIS SHOULD BE PRETRAIN LOSS AGAIN.
    indirect_grad = gather_flat_grad(
        grad(d_train_loss_d_w, hyper_params, grad_outputs=preconditioner.view(-1)))
    hypergrad = indirect_grad # + direct_Grad

    zero_hypergrad(hyper_params)
    store_hypergrad(hyper_params, -hypergrad)
    return hypergrad

nt_xent_criterion = NTXentLoss(device, args.batch_size, args.temperature, use_cosine_similarity=True)
def get_loss_simclr(student, xis, xjs):
    
    # print(xis.type(), xjs.type())
    # get the representations and the projections
    ris, zis = student(xis)  # [N,C]

    # get the representations and the projections
    rjs, zjs = student(xjs)  # [N,C]

    # normalize projection feature vectors
    # zis = F.normalize(zis, dim=1)
    # zjs = F.normalize(zjs, dim=1)

    loss = nt_xent_criterion(zis, zjs)
    return loss

def do_pretrain(student, head, teacher, optimizer, xis, xjs):
    student.train()
    xis = xis.to(device)
    xjs = xjs.to(device)
    if teacher is not None:
        xis = teacher(xis)
        xjs = teacher(xjs)
    loss = get_loss_simclr(student, xis, xjs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def inner_loop_finetune(student, head, teacher, optimizer, train_dl, val_dl, num_steps):
    stud_loss = 0.
    stud_acc = 0.

    student.eval()
    if teacher is not None:
        teacher.train()
    
    for i, (x,y) in enumerate(train_dl):
        x = x.to(device)
        y = y.to(device)
        y_loss, acc = get_loss(student, head, teacher, x, y)
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
    for i, (x,y) in enumerate(val_dl):
        x = x.to(device)
        y = y.to(device)
        y_loss, acc = get_loss(student, head,  teacher, x, y)

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

def do_ft_head(student, head, optimizer, dl):
    student.eval()
    for x,y in dl:
        x = x.to(device)
        y = y.to(device)

        loss, acc = get_loss(student,  head, None, x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        break
    return loss.item(), acc


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

def train():
    pt_meter = AverageMeter()
    ft_loss_meter = AverageMeter()
    ft_acc_meter = AverageMeter()

    # create save path
    if args.save:
        save_path = get_save_path()
        

    if args.teacherarch == 'warpexmag':
        teacher = RandWarpAugLearnExMag(inshape=[1024]).to(device)
        hyp_params = list(teacher.parameters())
        
        hyp_optim = torch.optim.Adam([
                {'params': teacher.net.parameters(), 'lr': args.hyper_lr},
                {'params': teacher.flow_mag_layer.parameters(), 'lr': args.hyper_lr},
                {'params': [teacher.flow_mag], 'lr': 1}])
        
        hyp_scheduler = None
    else:
        args.teacherarch = None
        teacher = None
        hyp_params = None
        hyp_optim = None
        hyp_scheduler = None

    DSHandle = ECGDataSetWrapper(args.batch_size)
    pretrain_dl, train_dl, val_dl, test_dl, _, NUM_TASKS_FT = DSHandle.get_data_loaders(args)

    if args.studentarch == 'resnet18':
        student = ecg_simclr_resnet18().to(device)
        head = MultitaskHead(256, NUM_TASKS_FT).to(device)
    else:
        raise NotImplementedError

    
    pretrain_optim = torch.optim.Adam(student.parameters(), lr=args.pretrain_lr)
    pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pretrain_optim, T_max=args.epochs, eta_min=0,
                                                               last_epoch=-1)
    finetune_optim = torch.optim.Adam(head.parameters(), lr=args.finetune_lr)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        student.load_state_dict(ckpt['student_sd'])
        if teacher is not None and ckpt['teacher_sd'] is not None: 
            teacher.load_state_dict(ckpt['teacher_sd'])
        head.load_state_dict(ckpt['head_sd'])
        pretrain_optim.load_state_dict(ckpt['pt_opt_state_dict'])
        pretrain_scheduler.load_state_dict(ckpt['pt_sched_state_dict'])
        finetune_optim.load_state_dict(ckpt['ft_opt_state_dict'])
        if teacher is not None and ckpt['hyp_opt_state_dict'] is not None: 
            hyp_optim.load_state_dict(ckpt['hyp_opt_state_dict'])
        load_ep = int(os.path.split(args.checkpoint)[-1][16:-3]) + 1
        print(f"Restored from epoch {load_ep}")
    else:
        print("Training from scratch")
        load_ep = 0
    
    if args.teach_checkpoint:
        print("LOADING PT AUG MODEL")
        ckpt = torch.load(args.teach_checkpoint)
        teacher.load_state_dict(ckpt['aug_sd'])
        print("LOAD SUCCESSFUL")

    stud_pretrain_ld = {'loss' : [], 'acc' : [] }
    stud_finetune_train_ld = {'loss' : [], 'acc' : []}
    stud_finetune_val_ld = {'loss' : [], 'acc' : []}
    stud_finetune_test_ld = {}

    num_finetune_steps = args.finetune_steps
    num_neumann_steps = args.neumann

    steps = 0
    for n in range(load_ep,args.epochs):
        # gradnorm = {'gradnorm' : []}
        progress_bar = tqdm(pretrain_dl)
        for i, (xis,xjs) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(n))
            # step_num = i + n*len(pretrain_dl)
            if teacher is not None:
                zero_hypergrad(hyp_params)

            if n < args.warmup_epochs or teacher is None:
                pt_loss = do_pretrain(student, head, teacher, pretrain_optim, xis, xjs)
                pt_meter.update(pt_loss)
                if teacher is not None:
                    ft_train_loss, ft_train_acc = do_ft_head(student, head, finetune_optim, train_dl)
                else:
                    ft_train_loss, ft_train_acc = 0,0
                ft_loss_meter.update(ft_train_loss)
                ft_acc_meter.update(ft_train_acc)
                ft_val_loss, ft_val_acc, hypg = 0,0,0
            else:
                # if steps % args.pretrain_steps == 0:
                pt_loss = do_pretrain(student, head, teacher, pretrain_optim, xis, xjs)
                pt_meter.update(pt_loss)
                if steps % args.pretrain_steps == 0:
                    with higher.innerloop_ctx(head, finetune_optim, copy_initial_weights=True) as (fnet, diffopt):
                        (ft_train_loss, ft_train_acc), (ft_val_loss, ft_val_acc), ft_grad, fnet = \
                                            inner_loop_finetune(student, fnet, teacher, diffopt, train_dl, val_dl, num_finetune_steps)
                        head.load_state_dict(fnet.state_dict())
                    ft_loss_meter.update(ft_train_loss)
                    ft_acc_meter.update(ft_train_acc)
                    ft_grad = gather_flat_grad(ft_grad)
                    for param_group in pretrain_optim.param_groups:
                        cur_lr = param_group['lr']
                        break
                    
                    hypg = hyper_step(student, head, teacher, hyp_params, pretrain_dl, pretrain_optim, ft_grad, cur_lr, num_neumann_steps)
                    hypg = hypg.norm().item()
                    hyp_optim.step()
                else:
                    ft_train_loss, ft_train_acc =  do_ft_head(student, head, finetune_optim, train_dl)
                    ft_loss_meter.update(ft_train_loss)
                    ft_acc_meter.update(ft_train_acc)
                    ft_val_loss, ft_val_acc, hypg = 0,0,0
            steps += 1
            progress_bar.set_postfix(
                    pretrain_loss='%.4f' % pt_meter.avg ,
                    finetune_train_loss='%.4f' % ft_loss_meter.avg ,
                    finetune_train_acc='%.4f' % ft_acc_meter.avg ,
                )

            # append to lossdict
            stud_pretrain_ld['loss'].append(pt_loss)
            stud_finetune_train_ld['loss'].append(ft_train_loss)
            stud_finetune_train_ld['acc'].append(ft_train_acc)
            stud_finetune_val_ld['loss'].append(ft_val_loss)
            stud_finetune_val_ld['acc'].append(ft_val_acc)

        if teacher is not None:    
            ft_test_ld = eval_student(student,head,  test_dl)
            stud_finetune_test_ld = update_lossdict(stud_finetune_test_ld, ft_test_ld)

            ft_val_ld = eval_student(student, head, val_dl)
            stud_finetune_val_ld = update_lossdict(stud_finetune_val_ld, ft_val_ld)

            ft_train_ld = eval_student(student,head, train_dl)
            stud_finetune_train_ld = update_lossdict(stud_finetune_train_ld, ft_train_ld)


        if hyp_scheduler is not None:
            hyp_scheduler.step()
        # reset the meter
        pt_meter.reset()
        ft_loss_meter.reset()
        ft_acc_meter.reset()
        # save the logs
        if args.save:
            tosave = {
                'pretrain_ld' : stud_pretrain_ld,
                'finetune_train_ld' : stud_finetune_train_ld,
                'finetune_val_ld' : stud_finetune_val_ld,
                'finetune_test_ld' : stud_finetune_test_ld,
            }
            torch.save(tosave, os.path.join(save_path, 'logs.ckpt'))
            if n == args.epochs- 1:
                model_saver(n, student, head, teacher, pretrain_optim, pretrain_scheduler, finetune_optim, hyp_optim, save_path)
                print(f"Saved model at epoch {n}")
    return student, head, teacher, pretrain_optim, pretrain_scheduler, finetune_optim, hyp_optim


res = train()








