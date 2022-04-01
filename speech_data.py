import torch
import torch.nn as nn
import os 
from torch.utils.data import Dataset
from data_tools import scale_ou, scale_in
from torchvision import transforms
import numpy as np 

class SpeechDataset(Dataset):
    def __init__(self, path):
        self.mask = []
        self.img = []

        n_files = len(os.listdir(path))
        
        #Use 16-4=12 files for trainning 
        # if path == 'engDataset/Train/spectrogram/':
        #   n_files -= 4

        for i in range(int(n_files/2)):
          img = np.load(path + 'noisy_voice_amp_db_{}'.format(i) + ".npy")
          clean_voice = np.load(path + 'voice_amp_db_{}'.format(i) + '.npy')
          mask = img - clean_voice

          img = scale_in(img)
          mask = scale_ou(mask)

          self.img.extend(img)
          self.mask.extend(mask)

          print(len(self.mask))
          print(len(self.img))
          print('*'*10)

        # self.img = np.array(self.img)
        # self.mask = np.array(self.mask)

        # self.img = scale_in(self.img)
        # self.mask = scale_ou(self.mask)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img = self.transform(self.img[index])
        mask = self.transform(self.mask[index])

        return img, mask

# PATH = 'engDataset/Val/spectrogram/'
# a = SpeechDataset(PATH)
# print(len(a))