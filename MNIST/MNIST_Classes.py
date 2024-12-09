from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset

class MNISTtraining(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15), # random image rotation. this is used as data augmentation, can be changed around based on the data
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].reshape((28,28)).astype(np.uint8)
        label = self.labels[index]
        idx = self.indices[index]
        image = self.transform(image)
        return {'image': image, 'label': label, 'index': idx}



class MNISTvalidation(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].reshape((28,28)).astype(np.uint8)
        label = self.labels[index]
        idx = self.indices[index]
        image = self.transform(image)
        return {'image': image, 'label': label, 'index': idx}


# wont have any labels as that is what we are trying to predict
class MNISTsubmit(Dataset):
    def __init__(self, images, indices):
        self.images = images
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].reshape((28,28)).astype(np.uint8)
        idx = self.indices[index]
        image = self.transform(image)
        return {'image': image, 'index': idx}