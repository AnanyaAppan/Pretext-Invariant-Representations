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
from module.corrflow import CorrFlow
import glob
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def resize_image(im, desired_size):
    old_size = im.shape  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size[:2])
    new_size = tuple([int(x*ratio) for x in old_size[:2]])
    print(new_size)
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    return new_im


def image_loader(path):
    image = cv2.imread(path)
    image = np.float32(image) / 255.0
    # image = resize_image(image,256)
    image = cv2.resize(image, (256, 256))
    return image

def quantized_color_preprocess(image, centroids):
    h, w, c = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    ab = image[:,:,1:]

    a = np.argmin(np.linalg.norm(centroids[None, :, :] - ab.reshape([-1,2])[:, None, :], axis=2),axis=1)
    # 256 256  quantized color (4bit)

    quantized_ab = a.reshape([h, w, -1])
    preprocess = transforms.ToTensor()
    return preprocess(quantized_ab)

def rgb_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(image)

class SSBDataset(Dataset):
    def __init__(self, video_path='../../SSBD/ssbd_clip_segment/'):
        self.video_path = video_path
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.centroids = np.load('./data/centroids_16k_kinetics_10000samples.npy',allow_pickle=True)
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
        # video_ret = []
        images_rgb = [] 
        images_quantized = []
        for img_name in video :
            image = image_loader(img_name)
            images_rgb.append(rgb_preprocess(image))
            images_quantized.append(quantized_color_preprocess(image,self.centroids))
            # image = torchvision.transforms.Resize((224, 224))(image)
            # image = torchvision.transforms.functional.to_tensor(image)
            # image = self.normalize(image)
            # video_ret.append(image)
        video_ret = [images_rgb,images_quantized]
        return (torch.stack(video_ret),label)

class ClassifyLSTM(nn.Module):
    def __init__(self, dr_rate=0.5):
        super(ClassifyLSTM, self).__init__()
        dr_rate = dr_rate
        rnn_hidden_size = 256
        rnn_num_layers = 1
        
        self.baseModel = CorrFlow().to(device)
        checkpoint = torch.load("../weights/corrflow.pth")['state_dict']
        model_dict = {}
        for key in checkpoint.keys() :
            model_dict[key[7:]] = checkpoint[key]
        self.baseModel.load_state_dict(model_dict)
        self.dropout = nn.Dropout(dr_rate)
        self.lstm_layer = nn.LSTM(128, 256, 10)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x_rgb, x_quantized):
        batch_sz, seq_len, c, h, w = x_rgb.shape
        ii = 0
        y = self.baseModel((x_rgb[:,ii]),(x_quantized[:,ii]),(x_rgb[:,ii+1]))
        print(y.shape)
        out, (hn, cn) = self.lstm_layer(y.unsqueeze(1))
        for ii in range(1, seq_len-1):
            y = self.baseModel((x_rgb[:,ii]),(x_quantized[:,ii]),(x_rgb[:,ii+1]))
            out, (hn, cn) = self.lstm_layer(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out 
        
lstm_model = ClassifyLSTM().to(device)
for name, param in lstm_model.named_parameters():
    if param.requires_grad:
        if 'baseModel' in name : param.requires_grad = False
optimizer = optim.SGD(lstm_model.parameters(), lr=1e-4, momentum=0.9)
# checkpoint = torch.load("../classify/epoch_49.pth")
# lstm_model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
criterion = nn.CrossEntropyLoss()
trainset = SSBDataset()
checkpoint_save_folder = "../classify_corrflow/"
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=8)
for epoch in range(50):
    print("Epoch = ", epoch)
    lstm_model.train()
    time1 = time.time()
    total = 0
    correct = 0
    running_loss = 0
    for i, data in enumerate(trainloader, 0) :
        videos, labels = data
        videos = videos.to(device)
        x_rgb, x_quantized = videos
        labels = labels.to(device)
        pred = lstm_model(x_rgb,x_quantized)
        step_loss = criterion(pred,labels)
        running_loss += step_loss.item()
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
    print('loss{}'.format(running_loss/len(trainloader)))
    print('accuracy{}'.format(100 * float(correct) / total))