import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image
import torch


class FoodDataset(Dataset):
    def __init__(self, datas, mode='train', cutout=False):
        self.datas = datas
        self.mode = mode
        mean = [117.40523108152614, 129.08895341687355, 138.8146819924039]
        std = [71.07845902847068, 67.67320379657173, 67.58323701160265]

        if mode == 'train':

            local_transform_list = [
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]

            if cutout:
                local_transform_list.append(Cutout(n_holes=2, length=4))

            self.local_transform = transforms.Compose(local_transform_list)

        elif mode == 'test':
            self.local_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            self.local_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        image, label = self.datas[idx]
        image = Image.fromarray(image)
        image = self.local_transform(image)

        return image, label


def get_training_dataloader(train_datas,
                            batch_size=16,
                            num_workers=0,
                            shuffle=True,
                            cutout=False):

    train_dataset = FoodDataset(train_datas, cutout=cutout)

    training_loader = DataLoader(train_dataset,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 batch_size=batch_size)

    return training_loader


def get_test_dataloader(test_datas, batch_size=16, num_workers=0):

    test_dataset = FoodDataset(test_datas, mode='test')
    test_loader = DataLoader(test_dataset,
                             num_workers=num_workers,
                             batch_size=batch_size)

    return test_loader


def compute_mean_std(dataset):

    data_r = numpy.dstack(
        [dataset[i][0][:, :, 0] for i in range(len(dataset))])
    data_g = numpy.dstack(
        [dataset[i][0][:, :, 1] for i in range(len(dataset))])
    data_b = numpy.dstack(
        [dataset[i][0][:, :, 2] for i in range(len(dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = numpy.ones((h, w), numpy.float32)

        for n in range(self.n_holes):
            y = numpy.random.randint(h)
            x = numpy.random.randint(w)

            y1 = numpy.clip(y - self.length // 2, 0, h)
            y2 = numpy.clip(y + self.length // 2, 0, h)
            x1 = numpy.clip(x - self.length // 2, 0, w)
            x2 = numpy.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


if __name__ == '__main__':
    datas = numpy.load('data/training_32.npz.npy', allow_pickle=True)
    # print(datas)
    print(datas.shape)
    # ((123.56389123633487, 135.426390337313, 154.83370455444475), (68.5118450943599, 66.29926692853485, 57.05538850824459))

    print(compute_mean_std(datas))
