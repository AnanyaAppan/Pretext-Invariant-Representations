import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from skimage import io
from PIL import Image
from jigsaw import Network
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SSBDataset(Dataset):
    def __init__(self, video_path='../../SSBD/ssbd_clip_segment/'):
        self.video_path = video_path
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._add_videos()

    def get_subset(self,arr):
        ret = []
        for i in range(len(arr)):
            if(i%5==0): ret.append(arr[i])
        return ret

    def _add_videos(self):
        self.arm_flapping_videos = [sorted(glob.glob(j + '/*.png'))[:100] for j in sorted(glob.glob(self.video_path+'ArmFlapping/*'))]
        self.head_banging_videos = [sorted(glob.glob(j + '/*.png'))[:100] for j in sorted(glob.glob(self.video_path+'HeadBanging/*'))]
        self.spinning_videos = [sorted(glob.glob(j + '/*.png'))[:100] for j in sorted(glob.glob(self.video_path+'Spinning/*'))]
        self.arm_flapping_videos = list(zip(self.arm_flapping_videos,[0]*len(self.arm_flapping_videos)))
        self.head_banging_videos = list(zip(self.head_banging_videos,[1]*len(self.head_banging_videos)))
        self.spinning_videos = list(zip(self.spinning_videos,[1]*len(self.spinning_videos)))
        self.videos = self.arm_flapping_videos + self.head_banging_videos + self.spinning_videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video, label = self.videos[idx]
        video_ret = []
        for img_name in video :
            image = Image.open(img_name)
            image = torchvision.transforms.Resize((224, 224))(image)
            image = torchvision.transforms.functional.to_tensor(image)
            image = self.normalize(image)
            video_ret.append(image)

        return (torch.stack(video_ret),label)

class ClassifyLSTM(nn.Module):
    def __init__(self, dr_rate=0.5):
        super(ClassifyLSTM, self).__init__()
        dr_rate = dr_rate
        rnn_hidden_size = 256
        rnn_num_layers = 1
        
        self.baseModel = Network().to(device)
        self.baseModel.train(False)
        self.baseModel.load_state_dict(torch.load("../jigsaw_models/epoch_585"))
        self.dropout = nn.Dropout(dr_rate)
        self.lstm_layer = nn.LSTM(128, 256, 1)
        self.fc1 = nn.Linear(256, 3)

    def forward(self, x):
        batch_sz, seq_len, c, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:,ii]))
        out, (hn, cn) = self.lstm_layer(y.unsqueeze(1))
        for ii in range(1, seq_len):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.lstm_layer(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out 
        
lstm_model = ClassifyLSTM().to(device)
optimizer = optim.SGD(lstm_model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
trainset = SSBDataset()
checkpoint_save_folder = "./classify/"
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=8)
total = 0
correct = 0
for epoch in range(50):
    print("Epoch = ", epoch)
    lstm_model.train()
    time1 = time.time()
    for i, data in enumerate(trainloader, 0) :
        videos, labels = data
        videos = videos.to(device)
        labels = labels.to(device)
        pred = lstm_model(videos)
        step_loss = criterion(pred,labels)
        # pred = torch.reshape(pred, (1, pred.shape[1]))
        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()
    time2 = time.time()
    save_name = os.path.join(checkpoint_save_folder, "epoch_{}.pth".format(epoch))  
    torch.save({
        'epoch': epoch,
        'model_state_dict': lstm_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': step_loss}, 
        save_name)
    print("saved_model...")
    checkpoint = torch.load(save_name)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    step_loss = checkpoint['loss']
    print("loaded model...")
    print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
    print('loss{}'.format(step_loss.item()))
    print('accuracy{}'.format(100 * float(correct) / total))