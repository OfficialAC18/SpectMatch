from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd 
import os
import torch.optim as optim
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
from torchmetrics.functional import f1_score
from sklearn.metrics import classification_report, mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score, rand_score, adjusted_rand_score, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from tqdm import tqdm
import numpy as np
logger = logging.getLogger(__name__)
scaler = GradScaler()

import models.AST as models
model = models.build_AST(num_classes = 2)

def test(test_loader, model, epoch):
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
    
    test_loader = tqdm(test_loader,disable=False)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            print("Iter No.", batch_idx)
            #inputs = inputs.transpose(1,3).to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_probs.append(outputs.detach().cpu().numpy())
            all_targets_tensor.append(targets)
            all_outputs_tensor.append(outputs)
            all_targets.append(targets.detach().cpu().numpy())
            all_outputs.append(np.argmax(outputs.detach().cpu().numpy(), axis = -1))
            loss = F.cross_entropy(outputs, targets)
            prec1, prec2 = accuracy(outputs, targets, topk=(1, 2))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top2.update(prec2.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            
            test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top2: {top2:.2f}.".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top2=top2.avg,
                ))
        
            test_loader.close()
            
    all_probs = np.concatenate(all_probs, axis = 0)
    all_targets = np.concatenate(all_targets, axis = 0)
    all_outputs = np.concatenate(all_outputs, axis = 0)
    all_targets_tensor = torch.cat(all_targets_tensor, dim = 0)
    all_outputs_tensor = torch.cat(all_outputs_tensor, dim = 0)
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-2 acc: {:.2f}".format(top2.avg))
    f1_macro = f1_score(all_outputs_tensor, all_targets_tensor, average = 'macro',num_classes = 2)
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

class Audioset(Dataset):
    def __init__(self,labels,dir_files,train=True): 
        super().__init__()
        if(train):
            self.targets = pd.read_csv(os.path.join(labels,'Train.csv')) 
        else:
            self.targets = pd.read_csv(os.path.join(labels,'Test.csv'))
        self.dir_files = dir_files
        self.train = train
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self,index):
        
        path = os.path.join(self.dir_files, str(self.targets.iloc[index,0])+'.npy')
        target = self.targets.iloc[index,1] - 2 #When Using Dataset-1, To Remove Offset
        target = torch.tensor(target)
        
        img = np.load(path)
        img = torch.tensor(img)
        img = torch.transpose(img,1,0)
        p = 298 - img.shape[0]
        
        if p > 0: 
            m = torch.nn.ZeroPad2d((0,0,0,p))
            img = m(img)
        elif p < 0:
            img = img[0:298, 0]
        
        return img,target
        
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging
import torch

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter']

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


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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
        


PATH = "/media/tower/Akchunya's Hardrive/Replay Attack Processed Data/AST/Oversampled"
TRAIN = "/media/tower/Akchunya's Hardrive/Replay Attack Processed Data/AST/Oversampled/Train/Labeled"
TEST = "/media/tower/Akchunya's Hardrive/Replay Attack Processed Data/AST/Oversampled/Test"
train_dataset = Audioset(labels = PATH, dir_files = TRAIN, train = True)
test_dataset = Audioset(labels = PATH, dir_files = TEST, train = False)

train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=25,
            num_workers=4,
            drop_last=True)
        

test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=3,
            num_workers=4)



device = torch.device('cuda', 0)
model = model.to(device)
local_rank = -1
out = '/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch-pytorch/CNN Run'
os.makedirs(out, exist_ok=True)
writer = SummaryWriter(out)

no_decay = ['bias', 'bn']
grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 5e-4},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
optimizer = optim.SGD(grouped_parameters, lr=0.001,momentum=0.9, nesterov=True)
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 2**20)

best_acc = 0
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


train_iter = iter(train_loader)
test_iter = iter(test_loader)

model.train()
for epoch in range(10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)
        
    p_bar = tqdm(range(13363), disable=False)
    for batch_idx in range(13363):
        try:
            inputs_x, targets_x = train_iter.next()
        except:
            train_iter = iter(train_loader)
            inputs_x, targets_x = train_iter.next()
            
        data_time.update(time.time() - end)
        batch_size = inputs_x.shape[0]
        targets_x = targets_x.to(device)
        inputs = inputs_x.float().to(device)
        #inputs = interleave(inputs,15).to(device)
        #inputs = inputs.transpose(1,3).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(inputs)
        #logits = de_interleave(logits,15)
        with torch.cuda.amp.autocast(enabled=True):
            loss = F.cross_entropy(logits, targets_x, reduction='mean')

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item())
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        
        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=10,
                    batch=batch_idx + 1,
                    iter=13363,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg))
        p_bar.update()


    
    p_bar.close()
    
    test_model = model
    
    test_loss, test_acc, test_f1, test_mf_score, test_adj_mf_score, test_norm_mf_score, test_rand_score, test_adj_rand_score, targets, outputs = test(test_loader, test_model, epoch)

    writer.add_scalar('train/1.train_loss', losses.avg, epoch)
    writer.add_scalar('test/1.test_acc', test_acc, epoch)
    writer.add_scalar('test/2.test_loss', test_loss, epoch)

    if(test_f1 > best_test_f1):
        torch.save(model.state_dict,'/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/results/remasc_AST@334053/model_state_dict.pt')
        torch.save(model,'/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/results/remasc_AST@334053/model.pt')
        torch.save(optimizer.state_dict,'/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/results/remasc_AST@334053/optimizer.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, '/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/results/remasc_AST@334053/checkpoint.pt')
        
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


    writer.close()


