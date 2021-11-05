"""Data loading utilities.
"""
import ipdb
import itertools
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision import datasets, transforms


def data_subset_indices(dataset, classes):
    try:
        if isinstance(dataset.targets, list):
            targets = torch.from_numpy(np.array(dataset.targets))
        else:
            targets = dataset.targets
        subset_indexes = torch.cat([torch.where(targets == c)[0] for c in classes])  # If the dataset has "targets"
    except:
        subset_indexes = np.concatenate([np.where(dataset.labels == c)[0] for c in classes])   # If the dataset has "labels"
    return subset_indexes


def get_cls_idx_dict(dataset):
    cls_to_idxs = {}
    try:
        if isinstance(dataset.targets, list):
            targets = torch.from_numpy(np.array(dataset.targets))
        else:
            targets = dataset.targets

        classes = torch.unique(targets).numpy()
        cls_to_idxs = { c: torch.where(targets == c)[0] for c in classes }  # If the dataset has "targets"
    except:
        try:
            classes = np.unique(dataset.labels)
            cls_to_idxs = { c: np.where(dataset.labels == c)[0] for c in classes }   # If the dataset has "labels"
        except:
            classes = np.unique(dataset)
            cls_to_idxs = { c: np.where(dataset == c)[0] for c in classes }
    return cls_to_idxs


def generate_occlusion_mask(t, occlusion_patch_size, image_size=32):
    patch_size = occlusion_patch_size
    rand_patch = np.random.rand(patch_size, patch_size)
    mask = np.array(Image.fromarray(rand_patch).resize((image_size, image_size), Image.BILINEAR)) > t
    return mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, classes=None, c_to_i=None, i_to_c=None, transform=None, target_transform=None,
                 corr_matrix=None, noise=0.0, occlusion_patch_size=4):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label
        self.corr_matrix = corr_matrix
        self.noise = noise
        self.occlusion_patch_size = occlusion_patch_size
        self.possible_classes = classes
        self.cls_idx_dict = get_cls_idx_dict(label)
        self.c_to_i = c_to_i
        self.i_to_c = i_to_c

    def __getitem__(self, index):
        possible_combos = list(itertools.product(self.possible_classes, self.possible_classes))
        combo_idx = np.random.choice(np.arange(len(possible_combos)), p=self.corr_matrix.reshape(-1))
        left_class, right_class = possible_combos[combo_idx]

        left_image_idx = np.random.choice(list(self.cls_idx_dict[left_class]))
        left_image = self.data[left_image_idx]
        left_image = self.process_image(left_image, self.noise, apply_noise=True)

        right_image_idx = np.random.choice(list(self.cls_idx_dict[right_class]))
        right_image = self.data[right_image_idx]
        right_image = self.process_image(right_image, self.noise, apply_noise=True)

        combined_image = np.concatenate([left_image, right_image], axis=2)
        label = np.array([self.c_to_i[left_class], self.c_to_i[right_class]])
        return combined_image, label

    def process_image(self, image, noise, apply_noise=True):
        if image.shape[0] != 1:
            image = Image.fromarray(np.uint8(np.asarray(image.transpose((1, 2, 0)))))
        elif image.shape[0] == 1:
            im = np.uint8(np.asarray(image))
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            image = Image.fromarray(im)

        if self.transform is not None:
            image = self.transform(image)

        image = np.array(image)
        gray_image = np.ones(image.shape) * 0.5
        mask = generate_occlusion_mask(noise, self.occlusion_patch_size)
        image = mask * image + (1 - mask) * gray_image
        return image.astype(np.float32)

    def __len__(self):
        return len(self.data)


class DatasetMulti(torch.utils.data.Dataset):
    def __init__(self, data, label, dim, classes=None, c_to_i=None, i_to_c=None, transform=None, target_transform=None,
                 corr=None, noise=0.0, occlusion_patch_size=4):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label
        self.corr = corr
        self.dim = dim
        self.noise = noise
        self.occlusion_patch_size = occlusion_patch_size
        self.possible_classes = classes
        self.cls_idx_dict = get_cls_idx_dict(label)
        self.c_to_i = c_to_i
        self.i_to_c = i_to_c

    def __getitem__(self, index):
        selected_classes = sample_z(1, self.corr, self.dim)[0]

        images = []
        for i in range(self.dim):
            image_idx = np.random.choice(list(self.cls_idx_dict[self.i_to_c[selected_classes[i]]]))
            image = self.data[image_idx]
            image = self.process_image(image, self.noise, apply_noise=True)
            images.append(image)

        combined_image = np.concatenate(images, axis=2)
        label = selected_classes
        return combined_image, label

    def process_image(self, image, noise, apply_noise=True):
        if image.shape[0] != 1:
            image = Image.fromarray(np.uint8(np.asarray(image.transpose((1, 2, 0)))))
        elif image.shape[0] == 1:
            im = np.uint8(np.asarray(image))
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            image = Image.fromarray(im)

        if self.transform is not None:
            image = self.transform(image)

        image = np.array(image)
        gray_image = np.ones(image.shape) * 0.5
        mask = generate_occlusion_mask(noise, self.occlusion_patch_size)
        # image = mask * image
        image = mask * image + (1 - mask) * gray_image
        return image.astype(np.float32)

    def __len__(self):
        return len(self.data)


def get_correlation_matrix(corr):
    c1 = 1 + corr
    c2 = 1 - corr
    corr_matrix = np.array([[c1, c2],
                            [c2, c1]])
    corr_matrix = corr_matrix / np.sum(corr_matrix)
    return corr_matrix


def get_correlated_data(dataset, classes, train_corr, test_corr, noise, occlusion_patch_size=4):
    NUM_TRAIN = 50000

    if dataset == 'mnist':
        # Train set
        trainset = datasets.MNIST(root='./data', train=True, download=True)
        trainset.data = trainset.data[:NUM_TRAIN, :, :]
        trainset.targets = trainset.targets[:NUM_TRAIN]
        # Validation set
        valset = datasets.MNIST(root='./data', train=True, download=True)
        valset.data = valset.data[NUM_TRAIN:, :, :]
        valset.targets = valset.targets[NUM_TRAIN:]
        # Test set
        testset = datasets.MNIST(root='./data', train=False, download=True)
    elif dataset == 'fashion_mnist':
        # Train set
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True)
        trainset.data = trainset.data[:NUM_TRAIN, :, :]
        trainset.targets = trainset.targets[:NUM_TRAIN]
        # Validation set
        valset = datasets.FashionMNIST(root='./data', train=True, download=True)
        valset.data = valset.data[NUM_TRAIN:, :, :]
        valset.targets = valset.targets[NUM_TRAIN:]
        # Test set
        testset = datasets.FashionMNIST(root='./data', train=False, download=True)

    c_to_i = {c: idx for (idx, c) in enumerate(classes)}
    i_to_c = {idx: c for (idx, c) in enumerate(classes)}

    subset_indices = data_subset_indices(trainset, classes=classes)
    train_data = trainset.data[subset_indices].unsqueeze(1).numpy()
    train_labels = trainset.targets[subset_indices]

    subset_indices = data_subset_indices(valset, classes=classes)
    val_data = valset.data[subset_indices].unsqueeze(1).numpy()
    val_labels = valset.targets[subset_indices]

    subset_indices = data_subset_indices(testset, classes=classes)
    test_data = testset.data[subset_indices].unsqueeze(1).numpy()
    test_labels = testset.targets[subset_indices]

    train_corr_matrix = get_correlation_matrix(train_corr)
    test_corr_matrix = get_correlation_matrix(test_corr)

    train_transform = transforms.Compose([transforms.Resize(32),
                                          transforms.ToTensor()])

    test_transform = transforms.Compose([transforms.Resize(32),
                                         transforms.ToTensor()])

    trainset = Dataset(train_data, train_labels, classes=classes, c_to_i=c_to_i, i_to_c=i_to_c, transform=train_transform,
                       corr_matrix=train_corr_matrix, noise=noise, occlusion_patch_size=occlusion_patch_size)
    valset = Dataset(val_data, val_labels, classes=classes, c_to_i=c_to_i, i_to_c=i_to_c, transform=train_transform,
                       corr_matrix=train_corr_matrix, noise=noise, occlusion_patch_size=occlusion_patch_size)
    testset = Dataset(test_data, test_labels, classes=classes,  c_to_i=c_to_i, i_to_c=i_to_c, transform=test_transform,
                      corr_matrix=test_corr_matrix, noise=noise, occlusion_patch_size=occlusion_patch_size)

    return trainset, valset, testset, c_to_i, i_to_c


def get_translation(dim):
    all_list = [[0, 1]] * dim
    return np.array(list(itertools.product(*all_list)))


def sample_z(num_samples, correlation, dim):
    """This version samples with given probabilites. In the future we might want to additionally
    add noise on z (in addition to x).
    """
    c1 = 1 + correlation
    c2 = 1 - correlation
    n = 2**(dim - 2)
    s2 = c2 / n
    s1 = c1 - (n-1) * s2

    if correlation >= 0:
        probabilities = np.ones(2**dim) * s2
        probabilities[0] = s1
        probabilities[-1] = s1
    elif correlation < 0:
        probabilities = np.ones(2**dim) * 1
        probabilities[0] = 0
        probabilities[-1] = 0

    probabilities /= sum(probabilities)
    samples = np.random.choice(np.arange(2**dim), size=num_samples, p=probabilities)
    translation = get_translation(dim)
    z = translation[list(samples)]
    return z


def get_correlated_data_multi(dataset, classes, dim, train_corr, test_corr, noise, occlusion_patch_size=4):
    NUM_TRAIN = 50000

    if dataset == 'mnist':
        # Train set
        trainset = datasets.MNIST(root='./data', train=True, download=True)
        trainset.data = trainset.data[:NUM_TRAIN, :, :]
        trainset.targets = trainset.targets[:NUM_TRAIN]
        # Validation set
        valset = datasets.MNIST(root='./data', train=True, download=True)
        valset.data = valset.data[NUM_TRAIN:, :, :]
        valset.targets = valset.targets[NUM_TRAIN:]
        # Test set
        testset = datasets.MNIST(root='./data', train=False, download=True)
    elif dataset == 'fashion_mnist':
        # Train set
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True)
        trainset.data = trainset.data[:NUM_TRAIN, :, :]
        trainset.targets = trainset.targets[:NUM_TRAIN]
        # Validation set
        valset = datasets.FashionMNIST(root='./data', train=True, download=True)
        valset.data = valset.data[NUM_TRAIN:, :, :]
        valset.targets = valset.targets[NUM_TRAIN:]
        # Test set
        testset = datasets.FashionMNIST(root='./data', train=False, download=True)

    c_to_i = {c: idx for (idx, c) in enumerate(classes)}
    i_to_c = {idx: c for (idx, c) in enumerate(classes)}

    subset_indices = data_subset_indices(trainset, classes=classes)
    train_data = trainset.data[subset_indices].unsqueeze(1).numpy()
    train_labels = trainset.targets[subset_indices]

    subset_indices = data_subset_indices(valset, classes=classes)
    val_data = valset.data[subset_indices].unsqueeze(1).numpy()
    val_labels = valset.targets[subset_indices]

    subset_indices = data_subset_indices(testset, classes=classes)
    test_data = testset.data[subset_indices].unsqueeze(1).numpy()
    test_labels = testset.targets[subset_indices]

    train_transform = transforms.Compose([transforms.Resize(32),
                                          transforms.ToTensor()])

    test_transform = transforms.Compose([transforms.Resize(32),
                                         transforms.ToTensor()])

    trainset = DatasetMulti(train_data, train_labels, dim=dim, classes=classes, c_to_i=c_to_i, i_to_c=i_to_c, transform=train_transform,
                            corr=train_corr, noise=noise, occlusion_patch_size=occlusion_patch_size)
    valset = DatasetMulti(val_data, val_labels, dim=dim, classes=classes, c_to_i=c_to_i, i_to_c=i_to_c, transform=train_transform,
                          corr=train_corr, noise=noise, occlusion_patch_size=occlusion_patch_size)
    testset = DatasetMulti(test_data, test_labels, dim=dim, classes=classes,  c_to_i=c_to_i, i_to_c=i_to_c, transform=test_transform,
                           corr=0.0, noise=noise, occlusion_patch_size=occlusion_patch_size)

    return trainset, valset, testset, c_to_i, i_to_c
