import logging
import math
from re import I
import warnings
warnings.simplefilter(action="ignore", category=UserWarning)
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchmetrics.functional import f1_score
import torchaudio
from torchvision.datasets import VisionDataset
import librosa
from sklearn.preprocessing import normalize
import torch
import cv2
import random
import os

from torchvision.transforms.transforms import ToPILImage


logger = logging.getLogger(__name__)
import numpy as np
from PIL import Image

def get_audiodset(args):
  transform_labeled = transforms.Compose([
        transforms.ToTensor(),
    ])

  transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])
  FOLDER = '/content/gdrive/MyDrive/NN Research/Dataset-1/Dataset-1 AST'
  TEST_PATH = '/content/gdrive/MyDrive/NN Research/Dataset-1/Dataset-1 AST/Test'
  TRAIN_LABELED_PATH = '/content/gdrive/MyDrive/NN Research/Dataset-1/Dataset-1 AST/Train/Labeled'
  TRAIN_UNLABELED_PATH = '/content/gdrive/MyDrive/NN Research/Dataset-1/Dataset-1 AST/Train/Unlabeled'
  class_labels = pd.read_csv(os.path.join(FOLDER,'Train.csv'))
  class_labels = class_labels.drop(class_labels.columns[[0]],axis = 1)
  class_labels = class_labels.labels


  train_labeled_idxs, train_unlabeled_idxs = x_u_split(
    args, class_labels)
  
  train_labeled_dataset = Audioset(labels = FOLDER, dir_files = TRAIN_LABELED_PATH,labeled=True,indexs = train_labeled_idxs,transform=transform_labeled)
  train_unlabeled_dataset = Audioset(labels = FOLDER, dir_files = TRAIN_UNLABELED_PATH, labeled = False, indexs = train_unlabeled_idxs, transform=TransformFixMatch())
  test_dataset = Audioset(labels = FOLDER, dir_files = TEST_PATH,transform=transform_labeled,train=False)

  return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    distinct_labels = np.unique(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in distinct_labels:
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, True)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx



class TransformFixMatch(object):
    def __init__(self, mean=None, std=None):
      self.freqm = torchaudio.transforms.FrequencyMasking(48)
      self.timem = torchaudio.transforms.TimeMasking(100)
  
    def __call__(self, weak, strong):
        weak_fbank = torch.tensor(weak)
        strong_fbank = torch.tensor(strong)

        weak_fbank = weak_fbank.unsqueeze(0)
        strong_fbank = strong_fbank.unsqueeze(0)

        #Performing Frequency and Time Masking
        weak_fbank = self.freqm(weak_fbank)
        strong_fbank = self.freqm(strong_fbank)
        weak_fbank = self.timem(weak_fbank)
        strong_fbank = self.timem(strong_fbank)

        weak_fbank = weak_fbank.squeeze(0)
        strong_fbank = strong_fbank.squeeze(0)
        weak_fbank = torch.transpose(weak_fbank, 0, 1)
        strong_fbank = torch.transpose(strong_fbank, 0, 1)
        #print('Shape of Weak fbank:',weak_fbank.shape)
        #print('Shape of Strong fbank:',strong_fbank.shape)


        return weak_fbank, strong_fbank

class Audioset(Dataset):
    def __init__(self,labels,dir_files,train=True,transform = None,target_transform = None,labeled=False,indexs=None): 
        super().__init__()
        self.labeled = labeled
        if(train):
            self.targets = pd.read_csv(os.path.join(labels,'Train.csv'))
            self.targets = self.targets.iloc[indexs]  
        else:
            self.targets = pd.read_csv(os.path.join(labels,'Test.csv'))
        self.dir_files = dir_files
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.mixup = 0
        self.noise = False
        self.target_length = 298
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self,index):

      if not (self.train and not self.labeled):
          path = os.path.join(self.dir_files, str(self.targets.iloc[index,2])[:-4]+'.npy')
          img = np.load(path)
          target = self.targets.iloc[index,1] - 2 #Removing Offset
          target = torch.tensor(target)
       
          if self.transform is not None:
            fbank = torch.tensor(img)
            fbank = torch.transpose(fbank, 1, 0)
            n_frames = fbank.shape[0]
            p = self.target_length - n_frames

            if p > 0:
              m = torch.nn.ZeroPad2d((0, 0, 0, p))
              fbank = m(fbank)
            elif p < 0:
              fbank = fbank[0:self.target_length, :]
            
            #print('Shape Of Labeled fbank:',fbank.shape)

        
          return fbank,target

      weak = np.load(os.path.join(self.dir_files +'/Weak',str(self.targets.iloc[index,2][:-4]+'.npy')))
      strong = np.load(os.path.join(self.dir_files +'/Strong',str(self.targets.iloc[index,2][:-4]+'.npy')))
      target = self.targets.iloc[index,1] - 2 #Removing Offset
      target = torch.tensor(target)

      if self.transform is not None:
        fbanks = self.transform(weak,strong)
  
      return fbanks,target


DSET_GETTERS_4 = {'classaudio_AST': get_audiodset}
                   

