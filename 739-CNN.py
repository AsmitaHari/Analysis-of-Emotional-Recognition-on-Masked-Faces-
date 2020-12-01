import os
import pandas as pd
import numpy as np
import torch
import glob
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import csv
import cv2
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 4
batch_size = 4
learning_rate = 0.001
class ImageLoader(Dataset):
    def __init__(self, dataset_file):
        self.label = list()
        self.dataset = list()
        try:
            with open(dataset_file, 'r') as f:
                csv_reader = csv.reader(f)
                for dat in csv_reader:
                    checkLen = False
                    x = np.array([])
                    for imagePath in glob.glob(dat[0]+"/*.png"):
                        dirName = os.path.dirname(imagePath)
                        length = len(glob.glob(dirName + "/*.png"))
                        if(length == 20):
                            checkLen = True
                            image = cv2.imread(imagePath)
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(gray, (224, 224))
                            resized = resized.reshape(1,224,224)
                            x = resized if x.size == 0 else np.vstack((x,resized))


                    if(checkLen):
                        self.label.append(torch.tensor([int(dat[1])]))
                        # np_array = np.array(dat[1:], dtype=np.float32).reshape(input_size, -1)
                        self.dataset.append(torch.from_numpy(x))
        except FileNotFoundError:
            print('generate features for [' + dataset_file + ']')
            exit(1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.label[idx]

    def to(self, device):
        for i in range(len(self.dataset)):
            self.label[i] = self.label[i].to(device=device)
            self.dataset[i] = self.dataset[i].to(device=device)
        return self
def lstm_style_batching(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    data = torch.cat(data, dim=1)
    label = torch.cat(label, dim=0)
    return data, label

def readImages():



    with open("imageFolder.csv", 'w',newline="") as f:
        writer = csv.writer(f)
        fileList = []
        for folder in glob.glob('C:/Users/asmit/PycharmProjects/pytorch/739-proj/outFolder/**/**'):
            row = []
            row.append(folder)
            fileName = os.path.basename(folder)
            dir = folder.split("\\")
            dirPath = os.path.join("C:/RIT-Stuff/Topics in System/Project/Emotion_labels/Emotion/", dir[1], dir[2])
            text = 0
            for filepath in glob.glob(dirPath + "/*.txt"):
                file1 = open(filepath, "r+")
                splitVal = file1.read().split(".")
                text = int(splitVal[0])
            row.append(text)
            fileList.append(row)
        writer.writerows(fileList)

        train_loader = DataLoader(ImageLoader("imageFolder.csv"), batch_size=2,shuffle=True,num_workers=8)

    return train_loader

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        ## formula = (w-f +2p)/s +1 w=input, f- filter size , p=padding, stride =1
        # self.conv1 = nn.Conv2d(20, 6, 5)  # 3 input channel RGB, 6 o/p channel, 5- filters
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)  ## for con2d o/p pf conv1 size is 6, 16 output size, 5 filters
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 120 can be any
        # self.fc2 = nn.Linear(120, 84)  # 84 can be anything
        # self.fc3 = nn.Linear(84, 7)  # 84 from prebious and 7 classes
        self.conv1 = nn.Conv2d(20, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 20)
        self.conv3 = nn.Conv2d(128, 256, 20)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(256)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2)
        # image size reduces by 2 while applying padding/ Hence last image will a dimesnsion 0f 16 * 5* 5

    def forward(self, x):
        # x= self.conv1(x.float())
        # x = self.pool(F.relu(x))

        # x = self.pool(F.relu(self.conv1(x.float())))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(1, x.size()[0], -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
        x = F.max_pool2d(F.relu(self.conv1(x.float())), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        x = x.view(1, x.size()[0], -1)
        x = self.adaptive_pool(x).squeeze()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
def train(train_loader):
    print("training")

    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimzer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, lables) in enumerate(train_loader):

            images = images.to(device)
            lables = lables.to(device)

            # forwaed
            output = model(images)
            loss = criterion(output, torch.max(lables, 1)[1])

            # backward
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()


            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss ={loss.item():.4f}')


if __name__ == '__main__':
    train_loader = readImages()
    print("Done with images")
    train(train_loader)