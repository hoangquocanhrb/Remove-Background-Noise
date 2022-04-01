from os import path
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats.stats import mode 
import torch
import torch.nn as nn
from torch._C import dtype
from torch.utils import data
from unet_resnet import UNET_RESNET
from unet_2 import UNET
from data_tools import scale_in, scale_ou
from scipy import stats
from torch.utils.data import DataLoader, dataloader, random_split
from speech_data import SpeechDataset

def training(train_path_save_spectrogram, val_path_save_spectrogram, model, device, epochs, batch_size):

    train_dataset = SpeechDataset(train_path_save_spectrogram)
    val_dataset = SpeechDataset(val_path_save_spectrogram)
    print('Length of train data : {}'.format(len(train_dataset)))
    print('Length of val dataset : ', len(val_dataset))

    data_loader = {}
    print('Get train loader')
    data_loader['train'] = DataLoader(train_dataset, batch_size=batch_size ,shuffle=True)
    print('Get val loader')
    data_loader['val'] = DataLoader(val_dataset, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.SmoothL1Loss()
    min_loss = 10

    for epoch in range(epochs):

        for phase in ['train', 'val']:
            print('Phase {} in epoch {} -----'.format(phase, epoch))
            if phase == 'train':
                model.train()
            if phase == 'val':
                model.eval()
            epoch_loss = 0
            
            for i, (imgs, mask) in enumerate(data_loader[phase]):
                imgs, mask = imgs.to(device, dtype=torch.float), mask.to(device, dtype=torch.float)

                with torch.set_grad_enabled(phase=='train'):
                    out_put = model(imgs)
                    loss = criterion(out_put, mask)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item()*imgs.size(0)
            epoch_loss /= len(data_loader[phase].dataset)
            print('Epoch {} loss {} = {}'.format(epoch, phase, epoch_loss))

            if phase == 'val':
              if epoch_loss < min_loss:
                torch.save(model.state_dict(), 'model/new_model3.pth')
                print('Saved model')
                min_loss = epoch_loss
        print('-'*15)

if __name__ == "__main__":
    train_path_save_spectrogram = 'engDataset/Train/spectrogram/'
    val_path_save_spectrogram = 'engDataset/Val/spectrogram/'
    model = UNET()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load('model/new_model2.pth', map_location=torch.device(device)))
    mode = model.to(device)
    training(train_path_save_spectrogram, val_path_save_spectrogram, model, device, 100, 90)
