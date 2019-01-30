from DenseNet import DenseNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import pandas as pd
from skimage import io, transform
import numpy as np
import cv2


class SinglePoseDataset(Dataset):

    def __init__(self, img_dir, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.img_dir = img_dir
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.images = datasets.ImageFolder(img_dir,
                         transforms.Compose([
                             transforms.Grayscale(),
                             transforms.ToTensor(),
                             normalize,
                         ]))

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        orientation = torch.tensor(self.data.iloc[idx, 1:].values.astype(dtype=np.float32))

        image = self.images[idx]
        sample = (image[0], orientation)

        return sample


def train(model, train_loader, optimizer, loss_func):
    model.train()
    num = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cuda"), target.to("cuda")
        optimizer.zero_grad()
        output = model(data)
        num += target.shape[0]
        loss = loss_func(output, target)
        total_loss += loss
        loss.backward()
        optimizer.step()

    print("Training Loss: " + str(total_loss.item()))


def test(model, test_loader, loss_func):
    model.eval()
    total_loss = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            total_loss += loss_func(output, target)
            num += output.shape[0]
        print("Test Loss: " + str(total_loss.item()))


def pose():
    batch_size = 48
    lr = 1e-3
    epochs = 100

    csv_path = "plane_data1/orientations.csv"
    image_dir = "plane_data1"
    pose_dataset = SinglePoseDataset(image_dir, csv_path)

    input_size = pose_dataset[0][0].shape[1]
    output_size = 9

    """train_split = 0.7
    train_size = int(train_split * len(pose_dataset))
    test_size = len(pose_dataset) - train_size
    """

    train_loader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=True)
    model = DenseNet(input_size=input_size, output_size=output_size, type="cont").cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    for epoch in range(1, epochs+1):
        train(model, train_loader, optimizer, loss_func)
        test(model, train_loader, loss_func)


if __name__ == '__main__':
    pose()











