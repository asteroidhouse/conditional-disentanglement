"""CelebA data loading functions.
"""
import os
import ipdb
import pandas
import getpass
import itertools
import collections
import numpy as np
from tqdm import tqdm

import matplotlib.pylab as plt

import torch
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.utils as vutils

# For optional occlusions
import data_utils


# CELEBA_ROOT = 'datasets/celeba'
# CELBEA_IMG_CACHE = os.path.join('data/celeba1k-64x64.npz')

CELEBA_ROOT = '/scratch/gobi1/datasets/celeba'
CELBEA_IMG_CACHE = os.path.join('/scratch/gobi1/pvicol/data/celeba1k-64x64.npz')

CELEBA_ATTRS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
                'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
                'Young']


def get_celeba_raw_attrs():
  df = pandas.read_csv(os.path.join(CELEBA_ROOT, 'Anno', 'list_attr_celeba.txt'),
                       skiprows=1, delimiter=r"\s+")
  df.index = df.index.map(lambda s: int(s.split('.')[0]) - 1)
  return df


def load_celeba_image_cache():
  if os.path.exists(CELBEA_IMG_CACHE):
    print("===> Loading cache")
    X = np.load(open(CELBEA_IMG_CACHE, 'rb'))
    X = torch.from_numpy(X.astype('float32') / 255.)
  else:
    tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((128)),
        transforms.Resize((64)),
        transforms.ToTensor()
    ])
    # Get Images
    print("=====> Loading all image")
    X = []
    celeba_imgdir = os.path.join(CELEBA_ROOT, 'Img', 'img_align_celeba')
    n_imgs = 202599
    for i in tqdm(range(1, n_imgs + 1)):
      im = plt.imread(os.path.join(celeba_imgdir, f"{i:06d}.jpg"))
      im = tr(im)
      X.append(im)
    X = torch.stack(X)
    print("DONE loading")

    # Cache
    if not os.path.exists('data'):
      os.makedirs('data')
    np.save(open(CELBEA_IMG_CACHE, 'wb'), (X.numpy() * 255).astype('uint8'))
  return X


def get_celeba_dataset(filter_variable, target_variable1, target_variable2):
  datasets = {}

  # Load Attributes
  df = get_celeba_raw_attrs()
  df[df == -1] = 0
  # Load Images
  X = load_celeba_image_cache()
  # Load Split
  split_df = pandas.read_csv(
      os.path.join(CELEBA_ROOT, 'Eval', 'list_eval_partition.txt'),
      delimiter=r"\s+",
      names=['filename', 'partition']
  )

  for split_idx, split_name in zip([0, 1, 2], ['train', 'val', 'test']):
    for f_val in [0, 1]:
      idx_filter = (df[filter_variable] == f_val) & (split_df['partition'] == split_idx)
      idxs = df[idx_filter].index.values
      Xs = X[idxs].float()
      a1 = df.iloc[idxs][target_variable1].values
      a2 = df.iloc[idxs][target_variable2].values
      As = np.hstack([a1.reshape(-1, 1), a2.reshape(-1, 1)])
      As = torch.from_numpy(As).long()
      datasets[f"{split_name}-{f_val}"] = torch.utils.data.TensorDataset(Xs, As)

      print('{} | f_val={}'.format(split_name, f_val))
      for a1_val, a2_val in itertools.product([0,1], [0,1]):
        print('(a1=={}, a2=={}): {}'.format(a1_val, a2_val, sum((a1==a1_val) & (a2==a2_val))))

  return datasets


class CorrelatedCelebaDataset(torch.utils.data.Dataset):
  def __init__(self, data, labels, corr_matrix=None, noise=0.0,
               occlusion_patch_size=4):
    self.data = data
    self.labels = labels
    self.noise = noise
    self.corr_matrix = corr_matrix
    self.occlusion_patch_size = occlusion_patch_size
    self.idx_dict = create_category_to_idxs_dict(labels)

  def __getitem__(self, index):
    possible_combos = list(itertools.product([0,1], [0,1]))
    combo_idx = np.random.choice(np.arange(len(possible_combos)),
                                 p=self.corr_matrix.reshape(-1))
    idx = np.random.choice(self.idx_dict[possible_combos[combo_idx]])
    image, label = self.data[idx], self.labels[idx]
    mask = data_utils.generate_occlusion_mask(self.noise,
                                              self.occlusion_patch_size,
                                              image_size=64)
    mask = torch.from_numpy(mask.astype(np.float32))
    image = mask * image
    return image, label

  def __len__(self):
    return len(self.data)


class CelebaDataset(torch.utils.data.Dataset):
  def __init__(self, data, labels, noise=0.0, occlusion_patch_size=4):
    self.data = data
    self.labels = labels
    self.noise = noise
    self.occlusion_patch_size = occlusion_patch_size

  def __getitem__(self, index):
    image, label = self.data[index], self.labels[index]
    mask = data_utils.generate_occlusion_mask(self.noise,
                                              self.occlusion_patch_size,
                                              image_size=64)
    mask = torch.from_numpy(mask.astype(np.float32))
    image = mask * image
    return image, label

  def __len__(self):
    return len(self.data)


def get_correlation_matrix(corr):
  c1 = 1 + corr
  c2 = 1 - corr
  corr_matrix = np.array([[c1, c2],
                          [c2, c1]])
  corr_matrix = corr_matrix / np.sum(corr_matrix)
  return corr_matrix


def get_count_matrix(labels):
  count_matrix = []
  for category in itertools.product([0,1], [0,1]):
    num_in_category = np.sum(np.all(labels == category, axis=1))
    count_matrix.append(num_in_category)
  count_matrix = np.array(count_matrix).reshape(2,2)
  return count_matrix


def get_adjusted_counts(count_matrix, corr_matrix):
  min_diag = min(count_matrix[0,0], count_matrix[1,1])
  min_offdiag = min(count_matrix[1,0], count_matrix[0,1])
  ratio = corr_matrix[0,0] / corr_matrix[0,1]

  while (min_diag / min_offdiag) > ratio:
    min_diag -= 1

  while (min_diag / min_offdiag) < ratio:
    min_offdiag -= 1

  adjusted_count_matrix = np.array([[min_diag, min_offdiag], [min_offdiag, min_diag]])
  return np.floor(adjusted_count_matrix).astype(np.int32)


def create_category_to_idxs_dict(labels):
  idx_dict = {}
  for category in itertools.product([0,1], [0,1]):
    idxs = np.where(np.all(labels == category, axis=1))[0]
    idx_dict[category] = idxs
  return idx_dict


def get_natural_correlated_celeba(factor1, factor2, splits=['train', 'val', 'test']):
  # Load Attributes
  df = get_celeba_raw_attrs()
  df[df == -1] = 0
  # Load Images
  X = load_celeba_image_cache()
  # Load Split
  split_df = pandas.read_csv(
      os.path.join(CELEBA_ROOT, 'Eval', 'list_eval_partition.txt'),
      delimiter=r"\s+",
      names=['filename', 'partition']
  )

  datasets = {}
  for split_idx, split_name in zip([0, 1, 2], ['train', 'val', 'test']):
    if split_name not in splits:
      continue

    idxs = df[split_df['partition'] == split_idx].index.values
    print('Split {} | Num examples {}'.format(split_name, len(idxs)))
    Xs = X[idxs].float()
    a1 = df.iloc[idxs][factor1].values
    a2 = df.iloc[idxs][factor2].values
    labels = np.hstack([a1.reshape(-1, 1), a2.reshape(-1, 1)])
    datasets[split_name] = CelebaDataset(Xs, labels)

  return datasets


def get_correlated_celeba(factor1, factor2, train_corr, test_corr,
                          splits=['train', 'val', 'test'], noise=0.0):
  train_corr_matrix = get_correlation_matrix(train_corr)
  test_corr_matrix = get_correlation_matrix(test_corr)

  # Load Attributes
  df = get_celeba_raw_attrs()
  df[df == -1] = 0
  # Load Images
  X = load_celeba_image_cache()
  # Load Split
  split_df = pandas.read_csv(
      os.path.join(CELEBA_ROOT, 'Eval', 'list_eval_partition.txt'),
      delimiter=r"\s+",
      names=['filename', 'partition']
  )

  datasets = {}
  for split_idx, split_name in zip([0, 1, 2], ['train', 'val', 'test']):
    if split_name not in splits:
      continue

    corr_matrix = train_corr_matrix if split_name in ['train', 'val'] else test_corr_matrix
    idxs = df[split_df['partition'] == split_idx].index.values
    print('Split {} | Num examples {}'.format(split_name, len(idxs)))
    Xs = X[idxs].float()
    a1 = df.iloc[idxs][factor1].values
    a2 = df.iloc[idxs][factor2].values
    labels = np.hstack([a1.reshape(-1, 1), a2.reshape(-1, 1)])
    idx_dict = create_category_to_idxs_dict(labels)
    count_matrix = get_count_matrix(labels)
    adjusted_count_matrix = get_adjusted_counts(count_matrix, corr_matrix)

    for i, (x,y) in enumerate(np.ndindex(adjusted_count_matrix.shape)):
      num_needed = adjusted_count_matrix[x,y]
      idx_dict[(x,y)] = np.random.choice(idx_dict[(x,y)], size=num_needed, replace=False)

    adjusted_idxs = np.concatenate(list(idx_dict.values()))
    Xs = Xs[adjusted_idxs]
    labels = labels[adjusted_idxs]
    datasets[split_name] = CelebaDataset(Xs, labels, noise=noise)

  return datasets


def get_correlated_celeba_sampled(factor1, factor2, train_corr, test_corr, noise=0.0):
  train_corr_matrix = get_correlation_matrix(train_corr)
  test_corr_matrix = get_correlation_matrix(test_corr)

  # Load Attributes
  df = get_celeba_raw_attrs()
  df[df == -1] = 0
  # Load Images
  X = load_celeba_image_cache()
  # Load Split
  split_df = pandas.read_csv(
      os.path.join(CELEBA_ROOT, 'Eval', 'list_eval_partition.txt'),
      delimiter=r"\s+",
      names=['filename', 'partition']
  )

  datasets = {}
  for split_idx, split_name in zip([0, 1, 2], ['train', 'val', 'test']):
    corr_matrix = train_corr_matrix if split_name == 'train' else test_corr_matrix
    idxs = df[split_df['partition'] == split_idx].index.values
    Xs = X[idxs].float()
    a1 = df.iloc[idxs][factor1].values
    a2 = df.iloc[idxs][factor2].values
    labels = np.hstack([a1.reshape(-1, 1), a2.reshape(-1, 1)])
    datasets[split_name] = CorrelatedCelebaDataset(Xs, labels, corr_matrix=corr_matrix, noise=noise)

  return datasets


def plot_joint_matrix(joint_matrix, factor1, factor2, title=None, fname=None):
  fig = plt.figure()
  im = plt.imshow(joint_matrix, cmap=plt.cm.inferno)

  yticks = ['{}=0'.format(factor2), '{}=1'.format(factor2)]
  xticks = ['{}=0'.format(factor1), '{}=1'.format(factor1)]

  plt.xticks([0, 1], yticks, fontsize=18)
  plt.yticks([0, 1], xticks, fontsize=18)

  # Rotate the tick labels and set their alignment.
  plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

  # Loop over data dimensions and create text annotations.
  for i in range(len(xticks)):
    for j in range(len(yticks)):
      text = plt.text(j, i, round(joint_matrix[i, j], 2), ha='center', va='center', color='g', fontsize=18)

  if title:
    plt.title(title, fontsize=20)

  plt.tight_layout()
  plt.savefig(os.path.join(fname), bbox_inches='tight', pad_inches=0)
  plt.close(fig)


if __name__ == '__main__':
  filter_variable = 'Attractive'
  factor1 = 'Male'
  factor2 = 'Black_Hair'
  correlation = 0.8
  noise = 0.4
  dataset_type = 'correlated2'

  if dataset_type == 'conditioned':
    datasets = get_celeba_dataset(filter_variable, factor1, factor2)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loaders = dict([(k, DataLoader(datasets[k],
                                   batch_size=100,
                                   shuffle=True if 'train' in k else False, **kwargs)) for k in datasets])
    # Plot the joint label distributions for the training and test sets to see how the filter variable affects them
    train_labels = loaders['train-0'].dataset.tensors[1].numpy()
    test_labels = loaders['val-1'].dataset.tensors[1].numpy()
  elif dataset_type == 'correlated1':
    datasets = get_correlated_celeba(factor1=factor1,
                                     factor2=factor2,
                                     train_corr=correlation,
                                     test_corr=-correlation,
                                     noise=noise)
    loaders = dict([(k, DataLoader(datasets[k],
                                   batch_size=100,
                                   shuffle=True if 'train' in k else False)) for k in datasets])
    # Plot the joint label distributions for the training and test sets to see how the filter variable affects them
    train_labels = loaders['train'].dataset.labels
    test_labels = loaders['val'].dataset.labels
  elif dataset_type == 'correlated2':
    datasets = get_correlated_celeba_sampled(factor1=factor1,
                                             factor2=factor2,
                                             train_corr=correlation,
                                             test_corr=-correlation,
                                             noise=noise)
    loaders = dict([(k, DataLoader(datasets[k],
                                   batch_size=100,
                                   shuffle=True if 'train' in k else False)) for k in datasets])
    train_labels = loaders['train'].dataset.labels
    test_labels = loaders['val'].dataset.labels

  train_matrix = []
  for category in itertools.product([0,1], [0,1]):
    num_in_category = np.sum(np.all(train_labels == category, axis=1))
    train_matrix.append(num_in_category)
  train_matrix = np.array(train_matrix).reshape(2,2)

  test_matrix = []
  for category in itertools.product([0,1], [0,1]):
    num_in_category = np.sum(np.all(test_labels == category, axis=1))
    test_matrix.append(num_in_category)
  test_matrix = np.array(test_matrix).reshape(2,2)

  plot_joint_matrix(train_matrix,
                    factor1=factor1,
                    factor2=factor2,
                    title='Train',
                    fname='joint_matrix_train.png')

  plot_joint_matrix(test_matrix,
                    factor1=factor1,
                    factor2=factor2,
                    title='Test',
                    fname='joint_matrix_test.png')

  # Plot count matrices when _sampling_ from the train and test sets
  # ----------------------------------------------------------------
  train_iterator = iter(loaders['train'])
  train_sampled_labels = []
  for i in range(100):
    images, labels = next(train_iterator)
    train_sampled_labels.append(labels.numpy())
  train_sampled_labels = np.concatenate(train_sampled_labels)

  train_sampled_matrix = []
  for category in itertools.product([0,1], [0,1]):
    num_in_category = np.sum(np.all(train_sampled_labels == category, axis=1))
    train_sampled_matrix.append(num_in_category)
  train_sampled_matrix = np.array(train_sampled_matrix).reshape(2,2)

  # Test set
  test_iterator = iter(loaders['test'])
  test_sampled_labels = []
  for i in range(100):
    images, labels = next(test_iterator)
    test_sampled_labels.append(labels.numpy())
  test_sampled_labels = np.concatenate(test_sampled_labels)

  test_sampled_matrix = []
  for category in itertools.product([0,1], [0,1]):
    num_in_category = np.sum(np.all(test_sampled_labels == category, axis=1))
    test_sampled_matrix.append(num_in_category)
  test_sampled_matrix = np.array(test_sampled_matrix).reshape(2,2)

  plot_joint_matrix(train_sampled_matrix,
                    factor1=factor1,
                    factor2=factor2,
                    title='Sampled Train',
                    fname='sampled_matrix_train.png')

  plot_joint_matrix(test_sampled_matrix,
                    factor1=factor1,
                    factor2=factor2,
                    title='Sampled Test',
                    fname='sampled_matrix_test.png')
  # ----------------------------------------------------------------

  if dataset_type == 'conditioned':
    for f_val in [0, 1]:
      x, a = loaders[f'train-{f_val}'].__iter__().__next__()
      vutils.save_image(x[:100], f'celeba-train-{f_val}.jpeg', normalize=True, nrow=10)
  else:
    x, a = loaders['train'].__iter__().__next__()
    vutils.save_image(x[:100], f'celeba-train.jpeg', normalize=True, nrow=10)

    x, a = loaders['test'].__iter__().__next__()
    vutils.save_image(x[:100], f'celeba-test.jpeg', normalize=True, nrow=10)
