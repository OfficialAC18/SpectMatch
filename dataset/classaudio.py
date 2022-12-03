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
import torchaudio
from torchvision.datasets import VisionDataset
import librosa
from sklearn.preprocessing import normalize
import torch
import cv2
import random
import os

from torchvision.transforms.transforms import ToPILImage

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)
import numpy as np
from PIL import Image

def get_audiodset(args):
  transform_labeled = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor()
        
    ])

  transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),
        
    ])
  FOLDER = '/content/gdrive/MyDrive/NN Research/Dataset-1/DataSet1/Audio_files'
  PATH = '/content/gdrive/MyDrive/NN Research/Dataset-1/DataSet1/Audio_files/audio_cleaned_3_seconds'
  class_labels = pd.read_csv(os.path.join(FOLDER,'Train.csv'))
  class_labels = class_labels.drop(class_labels.columns[[0]],axis = 1)
  class_labels = class_labels.labels


  train_labeled_idxs, train_unlabeled_idxs = x_u_split(
    args, class_labels)
  
  train_labeled_dataset = Audioset(labels = FOLDER, dir = PATH, labeled=True, train = True,indexs = train_labeled_idxs, transform=transform_labeled)
  train_unlabeled_dataset = Audioset(labels = FOLDER,dir = PATH, transform=TransformFixMatch(),indexs = train_unlabeled_idxs,train = True, labeled = False)
  test_dataset = Audioset(labels = FOLDER, dir=PATH,transform=transform_val,train=False, labeled=True)

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
        self.sr = 48000
        self.strong = [
                    ["lowpass", "-1", "300"],
                    ["speed", "0.8"],
                    ["rate", str(self.sr)],
                    ["reverb", "-w"],
                    ["rate",]
        ]
    
    def white_noise(self, signal, noise_factor = 0.3):
        noise = np.random.normal(0, signal.std(), signal.shape)
        noise *= noise_factor
        aug_signal = signal + noise
        return aug_signal

    def time_stretch(self,signal, stretch_rate = 0.4,sr = 48000):
        start_location = random.randint(1,4)
        aug_signal = librosa.effects.time_stretch(signal.reshape(-1), stretch_rate)
        aug_signal = aug_signal[start_location*sr:(start_location+3)*sr]
        return aug_signal

    
    def pitch_scaling(self,signal):
        num_semitones = random.randint(-8,8)
        return librosa.effects.pitch_shift(signal,48000,num_semitones) 

    def __call__(self, x):
        weak = self.pitch_scaling(self.time_stretch(self.white_noise(x)))
        strong,_ = torchaudio.sox_effects.apply_effects_tensor(torch.tensor(x.reshape(1,-1)), self.sr, self.strong, channels_first = True)
        strong = strong.numpy().reshape(-1)
        '''
        print("Shape of strong signal",strong.shape)
        '''
        #Calculating STFT for weak and strong signals
        stft_weak = np.abs(librosa.stft(weak, n_fft = 255, hop_length = 1125))
        stft_strong = np.abs(librosa.stft(strong, n_fft = 255, hop_length = 1125))

        #Calculating MFCC for weak and strong signals
        mfcc_weak = librosa.feature.mfcc(weak, sr = 48000, n_mfcc = 256, n_fft = 255, hop_length = 1125)
        mfcc_strong = librosa.feature.mfcc(strong, sr = 48000, n_mfcc = 256, n_fft = 255, hop_length = 1125)

        #Calculating Spec Centroid for weak and strong signals
        spec_centroid_weak = librosa.feature.spectral_centroid(y = weak, sr = 48000)
        spec_centroid_strong = librosa.feature.spectral_centroid(y = strong, sr = 48000)

        #Calculating Chroma STFT for weak and strong signals
        chroma_stft_weak = librosa.feature.chroma_stft(y = weak, sr = 48000)
        chroma_stft_strong = librosa.feature.chroma_stft(y = strong, sr = 48000)

        #Calcuating Spectral Bandwidth for weak and strong signals
        spec_bw_weak = librosa.feature.spectral_bandwidth(y = weak, sr = 48000)
        spec_bw_strong = librosa.feature.spectral_bandwidth(y = strong, sr = 48000)


        weak = np.array([normalize(spec_bw_weak)]).reshape(1,-1)
        strong = np.array([normalize(spec_bw_strong)]).reshape(1,-1)
        weak = np.append(weak, spec_centroid_weak, axis = 0)
        strong = np.append(strong, spec_centroid_strong, axis = 0) 

        for _ in range(0,9):
          weak = np.append(weak,normalize(spec_bw_weak), axis=0)
          strong = np.append(strong,normalize(spec_bw_strong), axis=0)
          weak = np.append(weak,normalize(spec_centroid_weak), axis=0)
          strong = np.append(strong,normalize(spec_centroid_strong), axis=0)
          weak = np.append(weak, normalize(chroma_stft_weak), axis=0)
          strong = np.append(strong, normalize(chroma_stft_strong), axis=0)

        weak = cv2.resize(weak,(128,128))
        weak = np.dstack((weak,np.abs(stft_weak),mfcc_weak))
        #weak = np.dstack((weak,mfcc_weak))

        strong = cv2.resize(strong,(128,128))
        stft_strong = cv2.resize(stft_strong,(128,128))
        mfcc_strong = cv2.resize(mfcc_strong,(128,128))
        '''
        print("Shape of strong:",strong.shape)
        print("Shape of stft:",stft_strong.shape)
        print("Shape of mfcc:",mfcc_strong.shape)
        '''
        strong = np.dstack((strong,np.abs(stft_strong),mfcc_strong))
        #image = np.dstack((image,mfcc))

        final_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),   
        ])

        weak = final_transforms(weak)
        strong = final_transforms(strong)

        return weak, strong

class Audioset(Dataset):
    def __init__(self,labels,dir,train=True,transform = None,target_transform = None,labeled=False,indexs=None): 
        super().__init__()
        self.labeled = labeled
        if(train):
            self.targets = pd.read_csv(os.path.join(labels,'Train.csv'))
            self.targets = self.targets.iloc[indexs]  
        else:
            self.targets = pd.read_csv(os.path.join(labels,'Test.csv'))
        self.dir = dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self,index):
        
        path = os.path.join(self.dir, str(self.targets.iloc[index,2]))
        
        signal, rate = librosa.load(path,sr = 48000)
        signal = signal.reshape(-1)
        target = self.targets.iloc[index,1] - 2 #Removing Offset
        target = torch.tensor(target)
        if not (self.train and not self.labeled):
          stft = np.abs(librosa.stft(signal, n_fft = 255, hop_length = 1125))
          mfcc = librosa.feature.mfcc(signal, sr = 48000, n_mfcc = 256, n_fft = 255, hop_length = 1125)
          spec_centroid = librosa.feature.spectral_centroid(y = signal, sr = 48000)
          chroma_stft = librosa.feature.chroma_stft(y = signal, sr = 48000)
          spec_bw = librosa.feature.spectral_bandwidth(y = signal, sr = 48000)


          image = np.array([normalize(spec_bw)]).reshape(1,-1)
          image = np.append(image, spec_centroid, axis = 0)

          for _ in range(0,9):
            image = np.append(image,normalize(spec_bw), axis=0)
            image = np.append(image,normalize(spec_centroid), axis=0)
            image = np.append(image, normalize(chroma_stft), axis=0)

          image = cv2.resize(image,(128,128))
          image = np.dstack((image,np.abs(stft),mfcc))
          #image = np.dstack((image,mfcc))
          #image = torch.tensor(image)
          image = Image.fromarray((image * 255).astype(np.uint8))

          if self.transform is not None:
            image = self.transform(image)
          

          return image, target


        #print("Shape of tensor:", torch.tensor(signal).shape)
        if self.transform is not None:
          signal = self.transform(signal)
        
        return signal,target


DSET_GETTERS = {'classaudio': get_audiodset}
                   

