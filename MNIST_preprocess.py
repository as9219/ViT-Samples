from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    


def MNIST_preprocess_data(train_csv, test_csv, random_seed):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=random_seed, shuffle=True)

    plt.figure()
    f, axarr = plt.subplots(1, 3)

    train_dataset = MNISTtraining(train_df.iloc[:, 1:].values.astype(np.uint8), train_df.iloc[:, 0].values, train_df.index.values)
    # print(len(train_dataset))
    # print(train_dataset[0])
    axarr[0].imshow(train_dataset[0]['image'].squeeze(), cmap='gray')
    axarr[0].set_title("Train Image")


    val_dataset = MNISTvalidation(val_df.iloc[:, 1:].values.astype(np.uint8), val_df.iloc[:, 0].values, val_df.index.values)
    # print(len(val_dataset))
    # print(val_dataset[0])
    axarr[1].imshow(val_dataset[0]['image'].squeeze(), cmap='gray')
    axarr[1].set_title('Validate Image')


    test_dataset = MNISTsubmit(test_df.values.astype(np.uint8), test_df.index.values) # dont need iloc as first column is not 'label'
    # print(len(test_dataset))
    # print(test_dataset[0])
    axarr[2].imshow(test_dataset[0]['image'].squeeze(), cmap='gray')
    axarr[2].set_title('Test Image')


    plt.show()

    return train_dataset, val_dataset, test_dataset