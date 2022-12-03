from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import pandas as pd 
import os
import torch.optim as optim
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
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
import onnxruntime as ort

import models.AST as models
model = models.build_AST(num_classes = 2)
device = torch.device('cuda')
model_onnx = ort.InferenceSession("/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/model_quant_4s.onnx")
torch.multiprocessing.set_sharing_strategy('file_system')

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()

def test(test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    end = time.time()
    all_outputs = []
    all_targets = []
    all_probs_onnx = []
    all_outputs_onnx = []
    all_targets_onnx = []
    all_probs = []
    all_targets_tensor = []
    all_outputs_tensor = []
    
    test_loader = tqdm(test_loader,disable=False)
    #dl = iter(t_loader)
    #dls = iter(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            #inputs_onnx, targets_onnx = dl.next()
            #inputs,targets = dls.next()
            data_time.update(time.time() - end)
            model.eval()
            print("Iter No.", batch_idx)
            #inputs = inputs.transpose(1,3).to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)
            #inp = {'input':to_numpy(inputs_onnx)}
            #out = model_onnx.run(None, inp)[0]
            #m = nn.Sigmoid()
            #out = m(torch.tensor(out.reshape(-1)))
            #all_probs_onnx.append(out.detach().cpu().numpy())
            #all_targets_onnx.append(targets_onnx)
            #all_outputs_onnx.append(np.argmax(out.detach().cpu().numpy(), axis=-1))
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
    #all_probs_onnx = np.concatenate(all_probs_onnx,axis = 0)
    #all_targets_onnx = np.concatenate(all_targets_onnx,axis = 0)
    #print(all_outputs_onnx)
    #print(all_probs_onnx.shape)
    #print(all_targets_onnx.shape)
    #all_outputs_onnx = np.array(all_outputs_onnx)
    
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
    '''
    print("*"*(len('Classification Report (ONNX)') + 2))
    print("*Classification Report (ONNX)*")
    print("*"*(len('Classification Report (ONNX)') + 2))
    print(classification_report(all_targets_onnx, all_outputs_onnx))
    '''
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
    np.save('/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/results/remasc_AST@334053/x_100_AST',all_probs)

    
    #return losses.avg, top1.avg, f1_macro, mf_score, adj_mf_score, norm_mf_score, rand__score, adj_rand_score, all_targets, all_probs

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
            img = img[0:298, 0:128]
        
        return img,target

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


PATH = "/media/tower/Akchunya's Hardrive/Replay Attack Processed Data/AST/Oversampled"
TEST = "/media/tower/Akchunya's Hardrive/Replay Attack Processed Data/AST/Oversampled/Test"
test_dataset = Audioset(labels = PATH, dir_files = TEST, train = False)
     
test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=3,
            num_workers=20)
t_loader = DataLoader(
            test_dataset,
            sampler = RandomSampler(test_dataset),
            batch_size=3,
            num_workers=20)


model.load_state_dict(torch.load('/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/results/remasc_AST@334053/checkpoint.pt')['model_state_dict'])
model.to(device)
test(test_loader,model)

