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
        train_mat = loadmat(join(data_path, 'devkit/cars_train_annos.mat'))
        train_lables = np.array([train_mat['annotations']['class'][0][i].item() for i in range(len(train_mat['annotations']['class'][0]))])
        train_images = np.array([join(data_path, f"cars_train/{train_mat['annotations']['fname'][0][i].item()}") for i in range(len(train_mat['annotations']['fname'][0]))])

        test_mat = loadmat(join(data_path, 'devkit/cars_test_annos_withlabels.mat'))
        test_lables = np.array([test_mat['annotations']['class'][0][i].item() for i in range(len(test_mat['annotations']['class'][0]))])
        test_images = np.array([join(data_path, f"cars_test/{test_mat['annotations']['fname'][0][i].item()}") for i in range(len(test_mat['annotations']['fname'][0]))])

        labels = np.concatenate((train_lables, test_lables))
        images = np.concatenate((train_images, test_images))

        # get split indices
        if split == 'train':
            selected_indices = np.where(labels <= 98)[0]
        elif split == 'val':
            selected_indices = np.where(labels > 98)[0]
        elif split == 'test':
            selected_indices = np.where(labels > 98)[0]
        else:
            raise NameError('undefined split')

        self.image_label = labels[selected_indices]
        self.image_list = images[selected_indices]

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
            positive = np.where(self.image_label == label)[0]                                      # find same-label samples
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
    def __init__(self, split='train', data_path='', margin=0.5, aug=True) -> None:
        super().__init__(split=split, data_path=data_path, aug=aug)

        self.margin = margin
        self.n_negative_subset = 1000

        # get positives
        self.positives = []
        for i, label in enumerate(self.image_label):
            positive = np.where(self.image_label == label)[0]                                      # find same-label samples
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
        if img.layers != 3:                      # some sample are greyscale images, which we should convert it to RGB by duplicating channels
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
        d_p, inds_p = dist.topk(1, largest=False)                                                  # choose the closet positive NOTE: choose the farthest positive?
        index_p = self.positives[index][inds_p].item()

        # mining the closet negative
        n_indices = self.negatives[index]
        # n_indices = np.random.choice(self.negatives[index], self.n_negative_subset)                # randomly choose potential_negatives
        n_emd = self.cache[n_indices]                                                              # (Np, D)
        dist = torch.norm(a_emd - n_emd, dim=1, p=None)                                            # Np
        d_n, inds_n = dist.topk(self.n_neg * 100, largest=False)                                   # choose the closet negative
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


#%%

if __name__ == '__main__':
    whole_train_set = Whole('train', data_path='dbs/CAR196', aug=True, return_label=True)
    whole_val_set = Whole('val', data_path='dbs/CAR196', aug=False, return_label=True)
    whole_test_set = Whole('test', data_path='dbs/CAR196', aug=False, return_label=True)
    print(f"length Test: {len(whole_test_set)}, cal: {int(0.2*len(whole_train_set))}, test: {int(0.8*len(whole_train_set))}")
