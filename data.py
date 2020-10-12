import numpy as np
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

MNIST_img_size = 28
MNIST_img_ch = 1
MNIST_num_classes = 10
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)    # 逗号是为了让他是tuple；如果没有逗号就是括号了，得到的值是int类型的


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.indices = np.arange(dataset_len)
        if shuffle:
            np.random.shuffle(self.indices)
        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        self.indices = tuple(self.indices.tolist())
    
    def __iter__(self):
        while True:
            p, q = 0, 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
    
    def __len__(self):
        return self.iters_per_ep


def get_dataloaders(data_root_path: str, batch_size: int):
    transform_train = tv.transforms.Compose([
        tv.transforms.RandomCrop(size=MNIST_img_size, padding=4),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    transform_test = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    
    train_set = tv.datasets.MNIST(
        root=data_root_path,
        train=True, download=True,  # 注意train参数传的值
        transform=transform_train
    )
    test_set = tv.datasets.MNIST(
        root=data_root_path,
        train=False, download=True, # 注意train参数传的值
        transform=transform_test
    )

    train_loader = DataLoader(
        dataset=train_set,
        pin_memory=True,
        num_workers=3,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(train_set),
            batch_size=batch_size,
            shuffle=True, drop_last=False
        )
    )
    test_loader = DataLoader(
        dataset=test_set,
        pin_memory=True,
        num_workers=3,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(test_set),
            batch_size=batch_size,
            shuffle=False, drop_last=False
        )
    )
    
    return train_loader, test_loader
