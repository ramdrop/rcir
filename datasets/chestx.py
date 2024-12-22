#%%
import numpy as np
from PIL import Image
from os.path import join, exists
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
from scipy.io import loadmat
import pandas as pd

name_to_num = {
    "Normal": 0,
    "Atelectasis": 1,
    "Calcification": 2,
    "Cardiomegaly": 3,
    "Consolidation": 4,
    "Diffuse Nodule": 5,
    "Effusion": 6,
    "Emphysema": 7,
    "Fibrosis": 8,
    "Fracture": 9,
    "Mass": 10,
    "Nodule": 11,
    "Pleural Thickening": 12,
    "Pneumothorax": 13,
}


def encode(labels):
    if len(labels) == 0:
        labels = ['Normal']
    label_compact = np.uint16(0)
    for label in labels:
        value = np.uint16(1) << name_to_num[label]
        label_compact = label_compact | value
    return label_compact



def decode(labels_compact):
    labels = []
    for i in range(13):
        if labels_compact & (np.uint16(1) << i):
            labels.append(i)
    return labels


# # test encode and decode
# label_toy = ['Atelectasis', 'Effusion', 'Nodule']
# label_toy = []
# label_compact = encode(label_toy)
# print(label_compact)
# print(decode(label_compact))


class Base(data.Dataset):
    def __init__(self, split='train', data_path='', aug=True) -> None:
        if aug:
            self.input_transform = T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.NEAREST, expand=False, center=None, fill=0),
                T.RandomResizedCrop((224, 224), scale=(0.4, 1.0), ratio=(0.75, 1.33), interpolation=T.InterpolationMode.NEAREST),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.input_transform = T.Compose([
                T.ToTensor(),
                T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
            ])

        df_train = pd.read_json(join(data_path, 'ChestX_Det_train.json'))
        df_test = pd.read_json(join(data_path, 'ChestX_Det_test.json'))
        train_raw_labels = df_train['syms'].values
        train_labels = []
        for labels in df_train['syms']:
            train_labels.append(encode(labels))
        train_images = np.array([join(data_path, 'all', x.replace('png', 'jpg')) for x in df_train['file_name']])
        train_labels = np.array(train_labels)

        test_labels = []
        for labels in df_test['syms']:
            test_labels.append(encode(labels))
        test_images = np.array([join(data_path, 'all', x.replace('png', 'jpg')) for x in df_test['file_name']])
        test_labels = np.array(test_labels)

        all_images = np.concatenate([train_images, test_images])
        all_labels = np.concatenate([train_labels, test_labels])

        # # see the number of samples in each class
        # for i in range(14):
        #     print(i, np.sum((all_labels & (np.uint16(1) << i))>0))

        train_labels, train_images, test_labels, test_images = [], [], [], []
        for i, label in enumerate(all_labels):
            if label & 0b110001111: # training set contains classes 0,1,2,3,7,8
                train_labels.append(all_labels[i])
                train_images.append(all_images[i])
            else:
                test_labels.append(all_labels[i])
                test_images.append(all_images[i])

        if split in ['train']:
            self.image_label = np.array(train_labels)
            self.image_list = np.array(train_images)
        elif split in ['val', 'test']:
            self.image_label = np.array(test_labels)
            self.image_list = np.array(test_images)


    def load_image(self, index):
        filepath = self.image_list[index]
        img = Image.open(filepath)
        if img.layers != 3:                      # some sample are greyscale images, which we should convert it to RGB by duplicating channels
            img = img.convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img


class Whole(Base):
    def __init__(self, split='train', data_path='', aug=True, return_label=False) -> None:
        super().__init__(split=split, data_path=data_path, aug=aug)

        self.return_label = return_label

        # get positives
        self.positives = []
        for i, label in enumerate(self.image_label):
            # positive = np.where(self.image_label == label)[0]                                      # find same-label samples
            positive = np.array(label & self.image_label).nonzero()[0]                                     # find same-label samples
            positive = np.delete(positive, np.where(positive == i)[0])                             # delete self
            self.positives.append(positive)

    def __len__(self):
        return len(self.image_list)

    def load_image(self, index):
        filepath = self.image_list[index]

        img = Image.open(filepath)
        if img.layers != 3:                      # some sample are greyscale images, which we should convert it to RGB by duplicating channels
            img = img.convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img

    def __getitem__(self, index):
        img = self.load_image(index)
        if self.return_label:
            label = self.image_label[index]
            return img, index, label
        else:
            return img, index

    def get_positives(self):
        return self.positives


class Tuple(Base):
    def __init__(self, split='train', data_path='',  margin=0.5, aug=True) -> None:
        super().__init__(split=split, data_path=data_path,  aug=aug)

        self.margin = margin
        self.n_negative_subset = 1000

        # get positives
        self.positives = []
        for i, label in enumerate(self.image_label):
            positive = np.where(label & self.image_label)[0]                                      # find same-label samples
            positive = np.delete(positive, np.where(positive == i)[0])                             # delete self
            self.positives.append(positive)

        # get negatives
        self.negatives = []
        for i, positive in enumerate(self.positives):
            negative = np.setdiff1d(np.arange(len(self.image_label)), positive, assume_unique=True)
            negative = np.delete(negative, np.where(negative == i)[0])                             # delete self
            self.negatives.append(negative)

        self.n_neg = 5
        self.cache = None                        # NOTE: assign a CPU tensor instead of a CUDA tensor to self.cache
        # self.cache = torch.zeros((len(self.image_label), 16))                                       # (N,D) dimension=16

    def __len__(self):
        return len(self.image_list)

    def load_image(self, index):
        filepath = self.image_list[index]
        img = Image.open(filepath)
        if img.layers != 3:                  # some sample are greyscale images, which we should convert it to RGB by duplicating channels
            img = img.convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img

    def __getitem__(self, index):

        # minig the closest positive
        p_indices = self.positives[index]
        # p_indices = np.random.choice(self.positives[index], int(0.5 * len(self.positives[index])), replace=False)
        a_emd = self.cache[index]                                                                  # (1,D)
        p_emd = self.cache[p_indices]                                                              # (Np, D)
        dist = torch.norm(a_emd - p_emd, dim=1, p=None)                                            # Np
        # print(dist.shape)
        d_p, inds_p = dist.topk(1, largest=False)                                                  # choose the closet positive NOTE: choose the farthest positive?
        index_p = self.positives[index][inds_p].item()

        # mining the closet negative
        n_indices = self.negatives[index]
        # n_indices = np.random.choice(self.negatives[index], self.n_negative_subset)                # randomly choose potential_negatives
        n_emd = self.cache[n_indices]                                                              # (Np, D)
        dist = torch.norm(a_emd - n_emd, dim=1, p=None)                                            # Np
        d_n, inds_n = dist.topk(self.n_neg * 100 if self.n_neg * 100 < len(dist) else len(dist), largest=False)                                   # choose the closet negative
        violating_indices = d_n < d_p + self.margin                                                # [True, True, ...] tensor
        if torch.sum(violating_indices) < 1:
            return None
        inds_n_vio = inds_n[violating_indices][:self.n_neg].numpy()                                # tensor -> numpy: a[tensor(5)] = 1, a[numpy(5)]=array([1])
        index_n = n_indices[inds_n_vio]

        # load images
        a_img = self.load_image(index)
        p_img = self.load_image(index_p)
        n_img = [self.load_image(ind) for ind in index_n]
        n_img = torch.stack(n_img, 0)            # (n_neg, C, H, W])

        return a_img, p_img, n_img, [index, index_p] + index_n.tolist()

    def get_positives(self):
        return self.positives


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)                                                 # ([B, C, H, W]) = ([C, H, W]) + ...
    positive = data.dataloader.default_collate(positive)                                           # ([B, C, H, W]) = ([C, H, W]) + ...
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])                   # ([B, C, H, W]) = ([C, H, W])
    negatives = torch.cat(negatives, 0)                                                            # ([B*n_neg, C, H, W])
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices


#%%

if __name__ == '__main__':

    # convert png to jpg
    import os
    from PIL import Image

    path = 'dbs/chest_x_det/all'
    for file in os.listdir(path):
        if file.endswith('.png'):
            im = Image.open(os.path.join(path, file))
            rgb_im = im.convert('RGB')
            rgb_im.save(os.path.join(path, file[:-4] + '.jpg'))
            # os.remove(os.path.join(path, file))