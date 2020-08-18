import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models

class CommaDataset(Dataset):
    def __init__(self, indices, speeds):
        self.speeds = speeds
        self.indices = indices
    
    def __getitem__(self, directory, index):
        prevIdx = self.indices[index]
        nextIdx = self.indices[index + 1]
        prevImage = directory + "/frame" + str(index) +".jpg"
        nextImage = directory + "/frame" + str(index + 1) +".jpg"
        flow_image = compute_optical_flow(prevImage, nextImage)
        return flow_image, self.speeds[prevIdx]
     
    def __len__(self):
        return len(self.indices)

class CommaModel(nn.Module):
    def __init__(self):
        super(CommaModel, self).__init__()
        self.backbone = int()
    
    def forward(self, x):
        return self.backbone(x)