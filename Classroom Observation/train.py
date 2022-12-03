import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import f1_score
from sklearn.metrics import classification_report, mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score, rand_score, adjusted_rand_score, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from tqdm import tqdm
from torch.cuda.amp import autocast,GradScaler

from dataset.cifar import DATASET_GETTERS
from dataset.classaudio import DSET_GETTERS
from dataset.classaudio_2 import DSET_GETTERS_2
from dataset.classaudio_spectrogram import DSET_GETTERS_3
from dataset.classaudio_AST import DSET_GETTERS_4
from dataset.classaudio_3_channels import DSET_GETTERS_5
from dataset.classaudio_AudioMAE import DSET_GETTERS_6
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0

scaler = GradScaler()

def get_all_roc_coordinates(y_real, y_proba):
  tpr_list = [0]
  fpr_list = [0]
  for i in range(len(y_proba)):
    threshold = y_proba[i]
    y_pred = y_proba >= threshold
    cm = confusion_matrix(y_real,y_pred)
    TN = cm[0,0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    tpr = TP/(TP+FN)
    fpr = 1-TN/(TN+FP)
    tpr_list.append(tpr)
    fpr_list.append(fpr)
  return tpr_list, fpr_list

def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
  sns.lineplot(x = fpr, y = tpr, ax = ax)
  sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
  plt.xlim(-0.05, 1.05)
  plt.ylim(-0.05, 1.05)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100','classaudio','classaudio_2','classaudio_spectrogram', 'classaudio_AST','classaudio_3_channels', 'classaudio_AudioMAE'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext', 'urban8k','ast','urban8k_v2','ViT','ViT-AudioMAE'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'urban8k':
            import models.audiocnn as models
            model = models.build_U8KCNN(num_classes = args.num_classes)
        
        elif args.arch == 'ast':
            import models.AST as models
            model = models.build_AST(num_classes = args.num_classes)
        
        elif args.arch == 'urban8k_v2':
            import models.audiocnn_v2 as models
            model = models.build_U8KCNN(num_classes = args.num_classes)
        elif args.arch == 'ViT':
            import models.ViT as models
            model = models.build_ViT(num_classes = args.num_classes)
        elif args.arch == 'ViT-AudioMAE':
            import models.ViT_AudioMAE as models
            model = models.build_ViTMAE(num_classes = args.num_classes)
            
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == 'classaudio' or args.dataset == 'classaudio_2' or args.dataset == 'classaudio_spectrogram' or args.dataset == 'classaudio_AST' or args.dataset == 'classaudio_3_channels' or args.dataset == 'classaudio_AudioMAE':
        args.num_classes = 5
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4 


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if(args.dataset != 'classaudio' and args.dataset != 'classaudio_2' and args.dataset != 'classaudio_spectrogram' and args.dataset != 'classaudio_AST' and args.dataset != 'classaudio_3_channels' and args.dataset != 'classaudio_AudioMAE'):
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')
    elif(args.dataset == 'classaudio'):
        labeled_dataset, unlabeled_dataset, test_dataset = DSET_GETTERS[args.dataset](
        args)
    elif(args.dataset == 'classaudio_2'):
        labeled_dataset, unlabeled_dataset, test_dataset = DSET_GETTERS_2[args.dataset](
        args)     
    elif(args.dataset == 'classaudio_spectrogram'):
        labeled_dataset, unlabeled_dataset, test_dataset = DSET_GETTERS_3[args.dataset](
        args)
    elif(args.dataset == 'classaudio_AST'):
        labeled_dataset, unlabeled_dataset, test_dataset = DSET_GETTERS_4[args.dataset](
        args)
    elif(args.dataset == 'classaudio_3_channels'):
        labeled_dataset, unlabeled_dataset, test_dataset = DSET_GETTERS_5[args.dataset](
        args)
    elif(args.dataset == 'classaudio_AudioMAE'):
        labeled_dataset, unlabeled_dataset, test_dataset = DSET_GETTERS_6[args.dataset](
        args)
     


    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    
    labeled_trainloader = DataLoader(
            labeled_dataset,
            sampler=train_sampler(labeled_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True)
        
    unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            sampler=train_sampler(unlabeled_dataset),
            batch_size=args.batch_size*args.mu,
            num_workers=args.num_workers,
            drop_last=True)

    test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers)

        


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    '''
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)
    '''  
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    '''      
    if args.amp:
        from apex import amp
    '''
    global best_acc
    best_test_f1 = 0
    low_mf_score = 1
    high_mf_score = 0
    low_adj_mf_score = 1
    high_adj_mf_score = 0
    low_norm_mf_score = 1
    high_norm_mf_score = 0
    low_rand_score = 1
    high_rand_score = 0
    low_adj_rand_score = 1
    high_adj_rand_score = 0
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            inputs = inputs.float()
            if args.amp:
              with torch.cuda.amp.autocast(enabled=True):
                logits = model(inputs)
            else:
              logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            if args.amp:
              with torch.cuda.amp.autocast(enabled=True):
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            else:
              Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            if args.amp:
              with torch.cuda.amp.autocast(enabled=True):
                Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            else:
                Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()
                                  
            loss = Lx + args.lambda_u * Lu


            if args.amp:
              optimizer.zero_grad()
              scaler.scale(loss).backward()
              scaler.step(optimizer)
              scaler.update()

              '''
              loss.backward()
              optimizer.step()
              scheduler.step()
              '''
            else:
              loss.backward()
              optimizer.step()
              scheduler.step()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_f1, test_mf_score, test_adj_mf_score, test_norm_mf_score, test_rand_score, test_adj_rand_score, targets, outputs = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            is_best_f1 = test_f1 > best_test_f1
            best_acc = max(test_acc, best_acc)
            best_test_f1 = max(test_f1, best_test_f1)
            high_mf_score = max(test_mf_score, high_mf_score)
            low_mf_score = min(test_mf_score, test_mf_score)
            high_adj_mf_score = max(test_adj_mf_score, high_adj_mf_score)
            low_adj_mf_score = min(test_adj_mf_score, low_adj_mf_score)
            high_norm_mf_score = max(test_norm_mf_score, high_norm_mf_score)
            low_norm_mf_score = min(test_norm_mf_score, low_norm_mf_score)
            high_rand_score = max(test_rand_score, high_rand_score)
            low_rand_score = min(test_rand_score, low_rand_score)
            high_adj_rand_score = max(test_adj_rand_score, high_adj_rand_score)
            low_adj_rand_score = min(test_adj_rand_score, low_adj_rand_score)


            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))
            logger.info('Best F1-Score (Macro): {:.2f}'.format(best_test_f1.item()))
            logger.info('Max Mutual Information Score : {:.4f}'.format(high_mf_score))
            logger.info('Min Mutual Information Score : {:.4f}'.format(low_mf_score))
            logger.info('Max Adjusted Mutual Information Score : {:.4f}'.format(high_adj_mf_score))
            logger.info('Min Adjusted Mutual Information Score : {:.4f}'.format(low_adj_mf_score))
            logger.info('Max Normalized Mutual Information Score: {:.4f}'.format(high_norm_mf_score))
            logger.info('Min Normalized Mutual Information Score: {:.4f}'.format(low_norm_mf_score))
            logger.info('Max Rand Score: {:.4f}'.format(high_rand_score))
            logger.info('Min Rand Score: {:.4f}'.format(low_rand_score))
            logger.info('Max Adjusted Rand Score: {:.4f}'.format(high_adj_rand_score))
            logger.info('Min Adjusted Rand Score: {:.4f}'.format(low_adj_rand_score))

            if(is_best_f1):
              #Saving the confusion matrix
              print('Outputs Shape:',outputs.shape)
              print('Targets Shape:',targets.shape)
              np.save(os.path.join(args.out,'Best Probs'),outputs)
              np.save(os.path.join(args.out,'Best Truths'),targets)
              conf_mat =  ConfusionMatrixDisplay.from_predictions(targets, np.argmax(outputs,axis = -1))
              plt.savefig(os.path.join(args.out,'Best Confusion Matrix.png'))

              #Getting the ROC Curves

              #Plotting OvR ROC Curves
              roc_auc_ovr_scores = {}
              plt.figure(figsize=(18,4))
              for i in range(4):
                ovr_actual = [1 if i==y else 0 for y in targets]
                ovr_prob = [prob[i] for prob in outputs]
                tpr,fpr = get_all_roc_coordinates(ovr_actual,ovr_prob)
                ax = plt.subplot(1,4,i+1)
                plot_roc_curve(tpr,fpr,scatter=False,ax=ax)
                ax.set_title(f'ROC Curve {i}vsR')
                roc_auc_ovr_scores[i] = roc_auc_score(ovr_actual,ovr_prob)

              
              plt.savefig(os.path.join(args.out,'Best ROC Curves OvR.png'))
              plt.close()
              '''

              with open('roc_auc_ovr.txt','w') as f:
                import json
                f.write(json.dumps(roc_auc_ovr_scores))
              '''

              #Constructing all required combinations
              class_combinations = []
              for i in range(4):
                for j in range(i+1,4):
                  class_combinations.append((i,j))
                  class_combinations.append((j,i))


              #Plotting OvO ROC Curves
              #Plotting OvO Curves and calculating their AUC
              roc_auc_ovo_scores = {}
              plt.figure(figsize=(20,20))
              for i in range(len(class_combinations)):
                  c1,c2 = class_combinations[i]
                  ovo_prob = [prob[c1] for prob, actual in zip(outputs,targets) if actual in [c1,c2]]
                  test_subset = [y_pred for y_pred in targets if y_pred in [c1,c2]]
                  test_subset = [1 if y==c1 else 0 for y in test_subset]
                  ax = plt.subplot(4,5,i+1)
                  tpr,fpr = get_all_roc_coordinates(test_subset,ovo_prob) 
                  plot_roc_curve(tpr,fpr,scatter=False,ax=ax)   
                  ax.set_title(f'ROC Curve {c1}vs{c2}')

      
                  #Calculating the AUC for the Curve
                  roc_auc_ovo_scores[(c1,c2)] = roc_auc_score(test_subset,ovo_prob)

              '''
                  #Saving the AUC values for the OvO curves
                  with open('roc_ovo.txt','w') as f:
                    import json
                    f.write(json.dumps(roc_auc_ovo_scores))
              '''

              plt.savefig(os.path.join(args.out,'Best ROC Curves OvO.png'))
              plt.close()

                



    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    end = time.time()
    all_outputs = []
    all_targets = []
    all_probs = []
    all_targets_tensor = []
    all_outputs_tensor = []

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            if (args.dataset == 'classaudio_3_channels'):
              inputs = inputs.to(args.device,dtype = torch.float)
            else:
              inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            all_probs.append(outputs.detach().cpu().numpy())
            all_targets_tensor.append(targets)
            all_outputs_tensor.append(outputs)
            all_targets.append(targets.detach().cpu().numpy())
            all_outputs.append(np.argmax(outputs.detach().cpu().numpy(), axis = -1))
            #outputs,indices = torch.max(outputs,dim=1)
            loss = F.cross_entropy(outputs, targets)
            prec1, prec2 = accuracy(outputs, targets, topk=(1, 2))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top2.update(prec2.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top2: {top2:.2f}.".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top2=top2.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    all_probs = np.concatenate(all_probs, axis = 0)
    all_targets = np.concatenate(all_targets, axis = 0)
    all_outputs = np.concatenate(all_outputs, axis = 0)
    all_targets_tensor = torch.cat(all_targets_tensor, dim = 0)
    all_outputs_tensor = torch.cat(all_outputs_tensor, dim = 0)
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-2 acc: {:.2f}".format(top2.avg))
    f1_macro = f1_score(all_outputs_tensor, all_targets_tensor, average = 'macro',num_classes = 5)
    mf_score = mutual_info_score(all_targets, all_outputs)
    adj_mf_score = adjusted_mutual_info_score(all_targets, all_outputs)
    norm_mf_score = normalized_mutual_info_score(all_targets, all_outputs)
    rand__score = rand_score(all_targets, all_outputs)
    adj_rand_score = adjusted_rand_score(all_targets, all_outputs)
    print('F1-score (Macro):', f1_macro.item())
    print("*"*(len('Classification Report') + 2))
    print("*Classification Report*")
    print("*"*(len('Classification Report') + 2))
    print(classification_report(all_targets, all_outputs))
    print("*"*(len('Mutual Information') + 2))
    print("*Mutual Information*")
    print("*"*(len('Mutual Information') + 2))
    print('Mutual Information Score: ',mf_score)
    print('Adjusted Mutual Information Score: ',adj_mf_score)
    print('Normalized Mutual Information Score: ', norm_mf_score)
    print("*"*(len('Rand_score') + 2))
    print("*Rand_score*")
    print("*"*(len('Rand_score') + 2))
    print('Rand Score: ',rand__score)
    print('Adjusted Rand Score: ',adj_rand_score)
    print("*"*(len('Confusion Matrix') + 2))
    print("*Confusion Matrix*")
    print("*"*(len('Confusion Matrix') + 2))


    
    return losses.avg, top1.avg, f1_macro, mf_score, adj_mf_score, norm_mf_score, rand__score, adj_rand_score, all_targets, all_probs


if __name__ == '__main__':
    main()
