"""Train on correlated real-world data (multi-digit MNIST or CelebA).

Examples
========

# Train on occluded 2-digit MNIST with conditional disentanglement
# ----------------------------------------------------------------
for NOISE in 0.0 0.2 0.4 0.6 0.8 ; do
    for TRAIN_CORRELATION in 0.0 0.6 0.9 ; do
        python train_correlated.py \
            --model=mlp \
            --epochs=400 \
            --dataset_type=multi-digit \
            --dataset=mnist \
            --train_corr=$TRAIN_CORRELATION \
            --test_corr=-$TRAIN_CORRELATION \
            --D_lr=1e-4 \
            --lr=1e-5 \
            --cls_lr=1e-4 \
            --z_dim=10 \
            --noise=$NOISE \
            --mi_type=conditional \
            --save_dir=saves/multi_mnist_cond &
    done
done

# Train on correlated Celeb-A with conditional disentanglement
# ------------------------------------------------------------
for TRAIN_CORRELATION in 0.0 0.2 0.4 0.6 0.8 ; do
    python train_correlated.py \
        --model=mlp \
        --epochs=200 \
        --dataset_type=correlated1 \
        --target_variable1=Male \
        --target_variable2=Smiling \
        --plot_covariance \
        --train_corr=$TRAIN_CORRELATION \
        --test_corr=-$TRAIN_CORRELATION \
        --D_lr=1e-4 \
        --lr=1e-5 \
        --cls_lr=1e-4 \
        --num_cls_steps=1 \
        --z_dim=10 \
        --disentangle_weight=10.0 \
        --mi_type=conditional \
        --save_dir=saves/celeba_cond &
done
"""
import os
import sys
import ipdb
import random
import argparse
import itertools
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict

from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import numpy as np
import sklearn.metrics
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

import torchvision

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('white')
sns.set_palette('muted')

from tensorboardX import SummaryWriter

import models
import resnet
import plot_utils
import data_utils
from logger import CSVLogger
from mmd import poly_mmd2, mix_rbf_mmd2, SIGMAS

import celeba
from celeba import CELEBA_ATTRS

from entropy_estimators import continuous


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='conditioned',
                    help='Dataset type to load (dummy, conditioned, correlated1, correlated2)')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='Which dataset to load (celeba, mnist, fashion)')
parser.add_argument('-f', '--filter_variable', type=str, choices=CELEBA_ATTRS)
parser.add_argument('-t1', '--target_variable1', type=str, choices=CELEBA_ATTRS,
                    help='First attribute name')
parser.add_argument('-t2', '--target_variable2', type=str, choices=CELEBA_ATTRS,
                    help='Second attribute name')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs')
parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'resnet'],
                    help='Model')
parser.add_argument('--nhid', type=int, default=50,
                    help='Number of hidden units per layer of the MLP')
parser.add_argument('--z_dim', type=int, default=10,
                    help='Dimension of z')

parser.add_argument('--f1_label_noise', type=float, default=0.0,
                    help='Amount of label noise to the first factor')
parser.add_argument('--f2_label_noise', type=float, default=0.0,
                    help='Amount of label noise to the second factor')
parser.add_argument('--train_corr', type=float, default=0.0,
                    help='Train-time correlation between the factors')
parser.add_argument('--test_corr', type=float, default=0.0,
                    help='Test-time correlation between the factors')
parser.add_argument('--noise', type=float, default=0.0,
                    help='Amount of observation noise (occlusions)')
parser.add_argument('--occlusion_patch_size', type=int, default=4,
                    help='Patch size used to generate occlusions')

parser.add_argument('--mi_type', type=str, default='none',
                    help='Type of MI to minimize')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--cls_lr', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--num_cls_steps', type=int, default=1,
                    help='Number of steps to train the classification heads')
parser.add_argument('--D_lr', type=float, default=1e-4,
                    help='Discriminator learning rate')
parser.add_argument('--D_beta1', type=float, default=0.9,
                    help='Adam beta1 hyperparameter for the discriminator(s)')
parser.add_argument('--D_beta2', type=float, default=0.999,
                    help='Adam beta2 hyperparameter for the discriminator(s)')
parser.add_argument('--mmd_full_sigmas', type=int, default=0,
                    help='0: uses only 1 scale of sigma =1, 1: uses multiple scales')
parser.add_argument('--disentangle_weight', type=float, default=1.0,
                    help='Weighting for the MMD/GAN losses')

parser.add_argument('--plot_covariance', action='store_true', default=False,
                    help='Whether to save covariance matrix visualizations of the latent space')
parser.add_argument('--k_neighbors', type=int, default=3,
                    help='Number of nearest neighbors to use for kNN-based MI estimation')
parser.add_argument('--eval_mi_every', type=int, default=1000,
                    help='Estimate MI and CMI every N iterations')
parser.add_argument('--plot_every', type=int, default=1000,
                    help='Plot iteration losses/accs every N iterations')
parser.add_argument('--log_every', type=int, default=10,
                    help='Log the full training and val losses to the CSV log every N iterations')
parser.add_argument('--prefix', type=str, default='',
                    help='Optional experiment name prefix')
parser.add_argument('--save_dir', type=str, default='saves',
                    help='Save directory')
parser.add_argument('--tb_logdir', type=str, default='runs',
                    help='(Optional) Directory in which to save Tensorboard logs')
parser.add_argument('--seed', type=int, default=3,
                    help='Random seed')
args = parser.parse_args()

use_device = 'cuda:0'

cudnn.benchmark = False
cudnn.deterministic = True

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

exp_name = 'ft1t2:{}_{}_{}-trnc:{}-tstc:{}-m:{}-lr:{}-clr:{}-dlr:{}-on:{}-z:{}-mi:{}-dl:{}-cls:{}-s:{}'.format(
            args.filter_variable, args.target_variable1, args.target_variable2, args.train_corr, args.test_corr,
            args.model, args.lr, args.cls_lr, args.D_lr, args.noise, args.z_dim, args.mi_type,
            args.disentangle_weight, args.num_cls_steps, args.seed)

if args.prefix:
    exp_name = args.prefix + '-' + exp_name

save_dir = os.path.join(args.save_dir, exp_name)

# Create experiment save directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, 'cov_plots'))
    os.makedirs(os.path.join(save_dir, 'loss_plots'))
    os.makedirs(os.path.join(save_dir, 'mi_plots'))
    os.makedirs(os.path.join(save_dir, 'z_space'))
    os.makedirs(os.path.join(save_dir, 'confusion'))
    os.makedirs(os.path.join(save_dir, 'prediction_vis'))
    os.makedirs(os.path.join(save_dir, 'prediction_vis_separate'))

# Save command-line arguments
with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
    yaml.dump(vars(args), f)

iteration_logger = CSVLogger(fieldnames=['epoch', 'global_iteration',
                                         'trn_loss', 'trn_f1_acc', 'trn_f2_acc',
                                         'val_loss', 'val_f1_acc', 'val_f2_acc',
                                         'tst_ac_loss', 'tst_ac_f1_acc', 'tst_ac_f2_acc',
                                         'tst_uc_loss', 'tst_uc_f1_acc', 'tst_uc_f2_acc'],
                             filename=os.path.join(save_dir, 'iteration.csv'))

if args.tb_logdir:
    writer = SummaryWriter(logdir=os.path.join(args.tb_logdir, exp_name))
else:
    writer = SummaryWriter(comment=exp_name)


classes = [0, 1]
c_to_i = {0: 0, 1: 1}
possible_labels = [c_to_i[v] for v in classes]

if args.dataset_type == 'conditioned':
    datasets = celeba.get_celeba_dataset(args.filter_variable, args.target_variable1, args.target_variable2)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loaders = dict([(k, DataLoader(datasets[k], batch_size=100, shuffle=True if 'train' in k else False, **kwargs)) for k in datasets])

    train_loader = loaders['train-0']
    test_loader = loaders['val-1']

elif args.dataset_type == 'correlated1':
    datasets = celeba.get_correlated_celeba(factor1=args.target_variable1,
                                            factor2=args.target_variable2,
                                            train_corr=args.train_corr,
                                            test_corr=args.test_corr,
                                            noise=args.noise,
                                            splits=['train', 'val', 'test'])
    train_loader = DataLoader(datasets['train'], batch_size=100, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=100, shuffle=True)
    test_loader_anticorrelated = DataLoader(datasets['test'], batch_size=100, shuffle=True)

    # This part is just to get the uncorrelated test set
    # --------------------------------------------------
    datasets_uncorrelated = celeba.get_correlated_celeba(factor1=args.target_variable1,
                                                         factor2=args.target_variable2,
                                                         train_corr=0.0,
                                                         test_corr=0.0,
                                                         noise=args.noise,
                                                         splits=['test'])
    test_loader_uncorrelated = DataLoader(datasets_uncorrelated['test'], batch_size=100, shuffle=True)
    # --------------------------------------------------

elif args.dataset_type == 'correlated2':
    datasets = celeba.get_correlated_celeba_sampled(factor1=args.target_variable1,
                                                    factor2=args.target_variable2,
                                                    train_corr=args.train_corr,
                                                    test_corr=args.test_corr,
                                                    noise=args.noise)
    train_loader = DataLoader(datasets['train'], batch_size=100, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=100, shuffle=True)
    test_loader = DataLoader(datasets['test'], batch_size=100, shuffle=True)

elif args.dataset_type == 'natural':
    datasets = celeba.get_natural_correlated_celeba(factor1=args.target_variable1,
                                                    factor2=args.target_variable2,
                                                    splits=['train', 'val', 'test'])
    train_loader = DataLoader(datasets['train'], batch_size=100, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=100, shuffle=True)
    test_loader_anticorrelated = DataLoader(datasets['test'], batch_size=100, shuffle=True)

    # This part is just to get the uncorrelated test set
    # --------------------------------------------------
    datasets_uncorrelated = celeba.get_correlated_celeba(factor1=args.target_variable1,
                                                         factor2=args.target_variable2,
                                                         train_corr=0.0,
                                                         test_corr=0.0,
                                                         noise=args.noise,
                                                         splits=['test'])
    test_loader_uncorrelated = DataLoader(datasets_uncorrelated['test'], batch_size=100, shuffle=True)
    # --------------------------------------------------

elif args.dataset_type == 'multi-digit':
    classes = [3, 8]
    trainset, valset, testset_anticorrelated, c_to_i, i_to_c = data_utils.get_correlated_data(args.dataset,
                                                                                              classes=classes,
                                                                                              train_corr=args.train_corr,
                                                                                              test_corr=args.test_corr,
                                                                                              noise=args.noise,
                                                                                              occlusion_patch_size=args.occlusion_patch_size)
    _, _, testset_uncorrelated, c_to_i, i_to_c = data_utils.get_correlated_data(args.dataset,
                                                                                classes=classes,
                                                                                train_corr=args.train_corr,
                                                                                test_corr=0,
                                                                                noise=args.noise,
                                                                                occlusion_patch_size=args.occlusion_patch_size)
    possible_labels = [c_to_i[v] for v in classes]
    train_loader = DataLoader(trainset, batch_size=100, shuffle=True)
    val_loader = DataLoader(valset, batch_size=100, shuffle=True)
    test_loader_anticorrelated = DataLoader(testset_anticorrelated, batch_size=100, shuffle=True)
    test_loader_uncorrelated = DataLoader(testset_uncorrelated, batch_size=100, shuffle=True)

# Plot heatmaps showing the number of examples for each combination of attributes
# -------------------------------------------------------------------------------
def add_border(image, color='red'):
    im = (image.transpose(1,2,0) * 255).astype(np.uint8)
    width, height = im.shape[0], im.shape[1]
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    draw.rectangle([(0,0), (width-1, height-1)], fill=None, outline=color, width=4)
    im = np.array(im).transpose(2,0,1).astype(np.float32) / 255.0
    return im


def visualize_predictions(dataloader, fname, M=50):
    annotated_image_list = []

    images, labels = next(iter(dataloader))  # Think about whether to change the structure of this to have a fixed minibatch always
    images = images.to(use_device)
    labels = labels.numpy()

    for i in range(len(images)):
        image, label = images[i], labels[i]
        image_np = image.cpu().numpy()

        z = model(image.unsqueeze(0))
        z1 = z[:,:subspace_dim]
        z2 = z[:,subspace_dim:]
        f1_logits = f1_classifier(z1)
        f2_logits = f2_classifier(z2)
        f1_pred = torch.argmax(f1_logits, axis=1).detach().cpu().item()
        f2_pred = torch.argmax(f2_logits, axis=1).detach().cpu().item()

        if (f1_pred != label[0]) and (f2_pred != label[1]):
            image_np = add_border(image_np, color='red')
        elif f1_pred != label[0]:
            image_np = add_border(image_np, color='orange')
        elif f2_pred != label[1]:
            image_np = add_border(image_np, color='purple')
        else:
            image_np = add_border(image_np, color='green')

        annotated_image_list.append(image_np)

    annotated_images_cat = np.stack(annotated_image_list, axis=0)[:50]
    grid = torchvision.utils.make_grid(torch.from_numpy(annotated_images_cat), nrow=5, pad_value=1)
    fig = plt.figure(figsize=(10,10))
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def visualize_predictions_separate(dataloader, fname_prefix, M=64):
    image_dict = defaultdict(list)

    for images, labels in dataloader:
        images = images.to(use_device)
        labels = labels.numpy()

        for i in range(len(images)):
            image, label = images[i], labels[i]
            image_np = image.cpu().numpy()

            z = model(image.unsqueeze(0))
            z1 = z[:,:subspace_dim]
            z2 = z[:,subspace_dim:]
            f1_logits = f1_classifier(z1)
            f2_logits = f2_classifier(z2)
            f1_pred = torch.argmax(f1_logits, axis=1).detach().cpu().item()
            f2_pred = torch.argmax(f2_logits, axis=1).detach().cpu().item()

            if (f1_pred != label[0]) and (f2_pred != label[1]):
                image_dict['all_incorrect'].append(image_np)
            elif f1_pred != label[0]:
                image_dict['f1_incorrect'].append(image_np)
            elif f2_pred != label[1]:
                image_dict['f2_incorrect'].append(image_np)
            else:
                image_dict['all_correct'].append(image_np)

        # Break out of the dataloader loop when we have enough examples in each category for the plots
        if all([len(image_dict[key]) >= M for key in image_dict]):
            break

    for key in image_dict:
        images_cat = np.stack(image_dict[key], axis=0)[:M]
        grid = torchvision.utils.make_grid(torch.from_numpy(images_cat), nrow=8, pad_value=1)
        fig = plt.figure(figsize=(10,10))
        plt.imshow(grid.numpy().transpose(1,2,0))
        plt.axis('off')
        plt.savefig('{}_{}.png'.format(fname_prefix, key), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def plot_joint_matrix(dataset, factor1=None, factor2=None, title=None, fname=None):
    possible_labels = [c_to_i[v] for v in classes]
    joint_matrix = []
    for category in itertools.product(possible_labels, possible_labels):
        num_in_category = np.sum(np.all(dataset.labels == category, axis=1))
        joint_matrix.append(num_in_category)
    joint_matrix = np.array(joint_matrix).reshape(2,2)

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

if args.dataset_type != 'multi-digit':
    plot_joint_matrix(train_loader.dataset,
                      factor1=args.target_variable1,
                      factor2=args.target_variable2,
                      title='Train c={}'.format(args.train_corr),
                      fname=os.path.join(save_dir, 'joint_matrix_train.png'))

    plot_joint_matrix(val_loader.dataset,
                      factor1=args.target_variable1,
                      factor2=args.target_variable2,
                      title='Val c={}'.format(args.train_corr),
                      fname=os.path.join(save_dir, 'joint_matrix_val.png'))

    plot_joint_matrix(test_loader_anticorrelated.dataset,
                      factor1=args.target_variable1,
                      factor2=args.target_variable2,
                      title='Test Anticorrelated c={}'.format(args.test_corr),
                      fname=os.path.join(save_dir, 'joint_matrix_test_anticorrelated.png'))

    plot_joint_matrix(test_loader_uncorrelated.dataset,
                      factor1=args.target_variable1,
                      factor2=args.target_variable2,
                      title='Test Uncorrelated',
                      fname=os.path.join(save_dir, 'joint_matrix_test_uncorrelated.png'))
# -------------------------------------------------------------------------------


# Save visualizations of the training and test data
# --------------------------------------------------------
train_images, train_labels = next(iter(train_loader))
train_grid = torchvision.utils.make_grid(train_images, nrow=8, pad_value=1)
fig = plt.figure(figsize=(10,10))
plt.imshow(train_grid.numpy().transpose(1,2,0))
plt.axis('off')
plt.savefig(os.path.join(save_dir, 'train_samples.png'), bbox_inches='tight', pad_inches=0)
plt.close(fig)

val_images, val_labels = next(iter(val_loader))
val_grid = torchvision.utils.make_grid(val_images, nrow=8, pad_value=1)
fig = plt.figure(figsize=(10,10))
plt.imshow(val_grid.numpy().transpose(1,2,0))
plt.axis('off')
plt.savefig(os.path.join(save_dir, 'val_samples.png'), bbox_inches='tight', pad_inches=0)
plt.close(fig)

test_images, test_labels = next(iter(test_loader_anticorrelated))
test_grid = torchvision.utils.make_grid(test_images, nrow=8, pad_value=1)
fig = plt.figure(figsize=(10,10))
plt.imshow(test_grid.numpy().transpose(1,2,0))
plt.axis('off')
plt.savefig(os.path.join(save_dir, 'test_anticorrelated_samples.png'), bbox_inches='tight', pad_inches=0)
plt.close(fig)

test_images, test_labels = next(iter(test_loader_uncorrelated))
test_grid = torchvision.utils.make_grid(test_images, nrow=8, pad_value=1)
fig = plt.figure(figsize=(10,10))
plt.imshow(test_grid.numpy().transpose(1,2,0))
plt.axis('off')
plt.savefig(os.path.join(save_dir, 'test_uncorrelated_samples.png'), bbox_inches='tight', pad_inches=0)
plt.close(fig)
# --------------------------------------------------------


# Training loop
# --------------------------------------------------------
num_f1_values = 2
num_f2_values = 2

subspace_dim = args.z_dim // 2

if args.model == 'mlp':
    if args.dataset_type == 'multi-digit':
        model = models.MLP(input_dim=1024*2*3, hidden_dim=args.nhid, output_dim=args.z_dim)
    else:
        model = models.MLP(input_dim=64*64*3, hidden_dim=args.nhid, output_dim=args.z_dim)
elif args.model == 'resnet':
    model = resnet.ResNet18(penultimate_size=2048, output_dim=args.z_dim)

f1_classifier = nn.Linear(subspace_dim, num_f1_values)  # Classifier to predict one of the three foreground digits
f2_classifier = nn.Linear(subspace_dim, num_f2_values)  # Classifier to predict one of the three background colors
model = model.to(use_device)
f1_classifier = f1_classifier.to(use_device)
f2_classifier = f2_classifier.to(use_device)
optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
classification_optimizer = optim.Adam(list(f1_classifier.parameters()) + list(f2_classifier.parameters()), lr=args.cls_lr)


pac_items = 10
class Discriminator_conditional():
    '''Discriminator for conditional loss
    '''
    def __init__(self, N, dim, lr, condition_dim, condition_value, z):
        self.dim = dim
        self.condition_dim = condition_dim
        self.condition_value = condition_value

        self.net = Net(self.dim).to(use_device)
        self.parameters = self.net.parameters()

        self.lr = lr
        self.optimizer_Discriminator = optim.Adam(list(self.parameters), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()

        self.z = z
        if self.condition_value == -1:
            self.mask = z[self.condition_dim] < 0.5
        elif self.condition_value == 1:
            self.mask = z[self.condition_dim] > 0.5

        self.N = self.mask.sum() // pac_items * pac_items

    def train(self, v, num_i_inner=1):
        v = v[:, self.mask]
        v = v[:, :self.N] # v must be dividable by pac_items
        running_loss_reals = 0
        running_loss_fakes = 0

        for i_inner in range(num_i_inner):
            # Shuffle v_real, such that there are different inputs stacked in each iteration:
            v_real = v[:, torch.randperm(self.N)]

            # Shuffle condition_dim to get v_fake
            v_fake = torch.clone(v_real)
            v_fake[self.condition_dim, :] = v_fake[self.condition_dim, torch.randperm(self.N)]

            # Take an optimisation step
            self.optimizer_Discriminator.zero_grad()
            prediction_reals = self.net(v_real).flatten()
            prediction_fakes = self.net(v_fake).flatten()
            label_reals = torch.ones(self.N // pac_items).to(use_device)
            label_fakes = torch.zeros(self.N // pac_items).to(use_device)
            loss_discriminator_reals = self.criterion(prediction_reals, label_reals)
            loss_discriminator_fakes = self.criterion(prediction_fakes, label_fakes)

            loss_discriminator = (loss_discriminator_reals + loss_discriminator_fakes) / 2
            loss_discriminator.backward()
            self.optimizer_Discriminator.step()

            # print intermediate steps
            running_loss_reals += loss_discriminator_reals.item()
            running_loss_fakes += loss_discriminator_fakes.item()
            if i_inner % 1000 == 999:    # print every 1000 iterations
                print('[%5d] loss: %.3f (%.3f)' %
                        (i_inner + 1, running_loss_reals / 1000, running_loss_fakes / 1000))
                running_loss_reals = 0.0
                running_loss_fakes = 0.0
        return ((loss_discriminator_reals + loss_discriminator_fakes) / 2).cpu().detach().numpy()

    def eval_fakes(self, v):
        '''This loss tests if Fakes look like Reals
        '''
        v = v[:, self.mask]
        v = v[:, :self.N]  # v must be dividable by pac_items
        v_fake = torch.clone(v)
        v_fake[self.condition_dim, :] = v_fake[self.condition_dim, torch.randperm(self.N)]
        prediction_fakes = self.net(v_fake).flatten()
        label_fakes = torch.ones(self.N // pac_items).to(use_device)
        return self.criterion(prediction_fakes, label_fakes)

    def eval_reals(self, v):
        '''This loss tests if Reals look like Fakes
        '''
        v = v[:, self.mask]
        v = v[:, :self.N]  # v must be dividable by pac_items
        prediction_reals = self.net(v).flatten()
        label_reals = torch.zeros(self.N // pac_items).to(use_device)
        return self.criterion(prediction_reals, label_reals)


if args.mi_type == 'unconditional':
    discriminator = models.MLP(input_dim=args.z_dim, hidden_dim=args.nhid, output_dim=1)
    discriminator = discriminator.to(use_device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=args.D_lr, betas=(args.D_beta1,args.D_beta2))
elif args.mi_type == 'conditional':
    # f1_discriminator = models.MLP(input_dim=args.z_dim + num_f1_values, hidden_dim=args.nhid, output_dim=1)  # 2 dims for X, 2 dims for concatenated one-hot class or domain label
    # f2_discriminator = models.MLP(input_dim=args.z_dim + num_f2_values, hidden_dim=args.nhid, output_dim=1)  # 2 dims for X, 2 dims for concatenated one-hot class or domain label
    # f1_discriminator = f1_discriminator.to(use_device)
    # f2_discriminator = f2_discriminator.to(use_device)

    dim = 2
    discr_list = []
    for d in range(dim):  # Number of latent subspaces
        discr_list.append(Discriminator_conditional(N, dim, lr_discr, d, -1, z=z_discret))
        discr_list.append(Discriminator_conditional(N, dim, lr_discr, d, 1, z=z_discret))

    # disc_optimizer = optim.Adam(list(f1_discriminator.parameters()) +
    #                             list(f2_discriminator.parameters()), lr=args.D_lr, betas=(args.D_beta1,args.D_beta2))


def evaluate(dataloader, save_name):
    losses = []

    # We want to keep track of how many predictions are correct
    # for each of the left and right-hand sides, separately for
    # each of the four combinations of left/right digits,
    # e.g., {(3,3), (3,8), (8,3), (8,8)}
    num_correct_f1_dict = defaultdict(int)
    num_correct_f2_dict = defaultdict(int)
    num_total_dict = defaultdict(int)

    num_correct_left = 0
    num_correct_right = 0
    num_total = 0

    all_z_np = []
    all_labels = []
    all_mapped_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            all_labels.append(labels.numpy())

            images = images.to(use_device)
            f1_labels = labels[:,0].to(use_device)
            f2_labels = labels[:,1].to(use_device)

            z = model(images)
            z1 = z[:,:subspace_dim]
            z2 = z[:,subspace_dim:]

            all_z_np.append(z.detach().cpu().numpy())

            f1_logits = f1_classifier(z1)
            f2_logits = f2_classifier(z2)

            f1_xentropy_loss = F.cross_entropy(f1_logits, f1_labels)
            f2_xentropy_loss = F.cross_entropy(f2_logits, f2_labels)
            loss = (f1_xentropy_loss + f2_xentropy_loss) / 2.0

            losses.append(loss.item())

            possible_labels = [c_to_i[v] for v in classes]
            f1_preds = torch.argmax(f1_logits, axis=1).detach().cpu().numpy()
            f2_preds = torch.argmax(f2_logits, axis=1).detach().cpu().numpy()
            concat_preds = np.concatenate([f1_preds.reshape(-1,1), f2_preds.reshape(-1,1)], axis=1)  # (100, 2)
            mapped_preds = np.ones(concat_preds.shape[0]) * -1  # Just initial array to hold results
            for (i, combo) in enumerate(itertools.product(possible_labels, possible_labels)):
                idxs = np.all(concat_preds == combo, axis=1)
                mapped_preds[idxs] = i
            all_mapped_preds.append(mapped_preds)

            for combo in itertools.product(possible_labels, possible_labels):
                idxs = torch.all(labels == torch.LongTensor(combo), axis=1)
                if not torch.all(idxs == False):
                    num_correct_f1_dict[combo] += torch.sum(torch.argmax(f1_logits[idxs], axis=1) == f1_labels[idxs]).item()
                    num_correct_f2_dict[combo] += torch.sum(torch.argmax(f2_logits[idxs], axis=1) == f2_labels[idxs]).item()
                    num_total_dict[combo] += torch.sum(idxs).item()

            num_correct_left += torch.sum(torch.argmax(f1_logits, axis=1) == f1_labels).item()
            num_correct_right += torch.sum(torch.argmax(f2_logits, axis=1) == f2_labels).item()
            num_total += len(images)

    z_np = np.concatenate(all_z_np)
    z_np -= z_np.mean(axis=0)  # Subtract mean
    z_np /= z_np.std(axis=0)   # Standardize
    covariance = (np.dot(z_np.T, z_np) / (z_np.shape[0]-1))  # Compute covariance

    if args.plot_covariance:
        fig = plt.figure()
        plt.imshow(covariance, cmap='seismic', vmin=-1, vmax=1)
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'cov_plots', 'full_cov_{}.png'.format(save_name)), bbox_inches='tight')
        plt.close(fig)

    left_acc = num_correct_left / float(num_total)
    right_acc = num_correct_right / float(num_total)

    f1_acc_dict = {combo: num_correct_f1_dict[combo] / float(num_total_dict[combo]) for combo in num_total_dict.keys()}
    f2_acc_dict = {combo: num_correct_f2_dict[combo] / float(num_total_dict[combo]) for combo in num_total_dict.keys()}

    all_z_np = np.concatenate(all_z_np)
    all_labels = np.concatenate(all_labels)
    all_mapped_preds = np.concatenate(all_mapped_preds)

    all_mapped_labels = np.ones(all_labels.shape[0]) * -1
    for (i, combo) in enumerate(itertools.product(possible_labels, possible_labels)):
        idxs = np.all(all_labels == combo, axis=1)
        all_mapped_labels[idxs] = i

    return np.mean(losses), left_acc, right_acc, f1_acc_dict, f2_acc_dict, all_z_np, all_labels, all_mapped_preds, all_mapped_labels


def save_models(prefix=None):
    if prefix:
        torch.save(model, os.path.join(save_dir, '{}-model.pt'.format(prefix)))
        torch.save(f1_classifier, os.path.join(save_dir, '{}-f1_classifier.pt'.format(prefix)))
        torch.save(f2_classifier, os.path.join(save_dir, '{}-f2_classifier.pt'.format(prefix)))
    else:
        torch.save(model, os.path.join(save_dir, 'model.pt'))
        torch.save(f1_classifier, os.path.join(save_dir, 'f1_classifier.pt'))
        torch.save(f2_classifier, os.path.join(save_dir, 'f2_classifier.pt'))


global_iteration = 0
iteration_dict = defaultdict(list)
epoch_dict = defaultdict(list)
mi_dict = defaultdict(list)

best_val_acc = 0.0
best_val_loss = 1e8

for epoch in range(args.epochs):

    # Evaluate mean loss and accuracy on the training and test sets (which differ in correlation between FG and BG
    trn_loss, trn_f1_acc, trn_f2_acc, trn_l_acc_dict, trn_r_acc_dict, trn_zs, trn_labels, trn_mapped_preds, trn_mapped_labels = evaluate(train_loader, save_name='train_ep_{}'.format(epoch))
    val_loss, val_f1_acc, val_f2_acc, val_l_acc_dict, val_r_acc_dict, val_zs, val_labels, val_mapped_preds, val_mapped_labels = evaluate(val_loader, save_name='val_ep_{}'.format(epoch))
    tst_ac_loss, tst_ac_f1_acc, tst_ac_f2_acc, tst_ac_l_acc_dict, tst_ac_r_acc_dict, tst_ac_zs, tst_ac_labels, tst_ac_mapped_preds, tst_ac_mapped_labels = evaluate(test_loader_anticorrelated, save_name='test_anticorr_ep_{}'.format(epoch))
    tst_uc_loss, tst_uc_f1_acc, tst_uc_f2_acc, tst_uc_l_acc_dict, tst_uc_r_acc_dict, tst_uc_zs, tst_uc_labels, tst_uc_mapped_preds, tst_uc_mapped_labels = evaluate(test_loader_uncorrelated, save_name='test_uncorr_ep_{}'.format(epoch))

    if epoch % 20 == 0:
        plot_utils.plot_confusion_matrix(trn_mapped_preds, trn_mapped_labels, 'Train Confusion', os.path.join(save_dir, 'confusion', 'train_confusion.png'))
        plot_utils.plot_confusion_matrix(val_mapped_preds, val_mapped_labels, 'Val Confusion', os.path.join(save_dir, 'confusion', 'val_confusion.png'))
        plot_utils.plot_confusion_matrix(tst_ac_mapped_preds, tst_ac_mapped_labels, 'Test Anticorrelated', os.path.join(save_dir, 'confusion', 'test_anticorr_confusion.png'))
        plot_utils.plot_confusion_matrix(tst_uc_mapped_preds, tst_uc_mapped_labels, 'Test Uncorrelated', os.path.join(save_dir, 'confusion', 'test_uncorr_confusion.png'))

        visualize_predictions(train_loader, os.path.join(save_dir, 'prediction_vis', 'train_preds.png'), M=50)
        visualize_predictions(val_loader, os.path.join(save_dir, 'prediction_vis', 'val_preds.png'), M=50)
        visualize_predictions(test_loader_anticorrelated, os.path.join(save_dir, 'prediction_vis', 'test_preds_anticorr.png'), M=50)
        visualize_predictions(test_loader_uncorrelated, os.path.join(save_dir, 'prediction_vis', 'test_preds_uncorr.png'), M=50)

        visualize_predictions_separate(train_loader, os.path.join(save_dir, 'prediction_vis_separate', 'train'), M=64)
        visualize_predictions_separate(val_loader, os.path.join(save_dir, 'prediction_vis_separate', 'val'), M=64)
        visualize_predictions_separate(test_loader_anticorrelated, os.path.join(save_dir, 'prediction_vis_separate', 'test_anticorr'), M=64)
        visualize_predictions_separate(test_loader_uncorrelated, os.path.join(save_dir, 'prediction_vis_separate', 'test_uncorr'), M=64)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        plot_utils.plot_confusion_matrix(trn_mapped_preds, trn_mapped_labels, 'Train Confusion', os.path.join(save_dir, 'confusion', 'train_confusion_bvl.png'))
        plot_utils.plot_confusion_matrix(val_mapped_preds, val_mapped_labels, 'Val Confusion', os.path.join(save_dir, 'confusion', 'val_confusion_bvl.png'))
        plot_utils.plot_confusion_matrix(tst_ac_mapped_preds, tst_ac_mapped_labels, 'Test Anticorrelated', os.path.join(save_dir, 'confusion', 'test_anticorr_confusion_bvl.png'))
        plot_utils.plot_confusion_matrix(tst_uc_mapped_preds, tst_uc_mapped_labels, 'Test Uncorrelated', os.path.join(save_dir, 'confusion', 'test_uncorr_confusion_bvl.png'))

        visualize_predictions_separate(train_loader, os.path.join(save_dir, 'prediction_vis_separate', 'train_bvl'), M=64)
        visualize_predictions_separate(val_loader, os.path.join(save_dir, 'prediction_vis_separate', 'val_bvl'), M=64)
        visualize_predictions_separate(test_loader_anticorrelated, os.path.join(save_dir, 'prediction_vis_separate', 'test_anticorr_bvl'), M=64)
        visualize_predictions_separate(test_loader_uncorrelated, os.path.join(save_dir, 'prediction_vis_separate', 'test_uncorr_bvl'), M=64)

        save_models(prefix='bvl')


    val_avg_acc = (val_f1_acc + val_f2_acc) / 2.0
    if val_avg_acc > best_val_acc:
        best_val_acc = val_avg_acc

        plot_utils.plot_confusion_matrix(trn_mapped_preds, trn_mapped_labels, 'Train Confusion', os.path.join(save_dir, 'confusion', 'train_confusion_bva.png'))
        plot_utils.plot_confusion_matrix(val_mapped_preds, val_mapped_labels, 'Val Confusion', os.path.join(save_dir, 'confusion', 'val_confusion_bva.png'))
        plot_utils.plot_confusion_matrix(tst_ac_mapped_preds, tst_ac_mapped_labels, 'Test Anticorrelated', os.path.join(save_dir, 'confusion', 'test_anticorr_confusion_bva.png'))
        plot_utils.plot_confusion_matrix(tst_uc_mapped_preds, tst_uc_mapped_labels, 'Test Uncorrelated', os.path.join(save_dir, 'confusion', 'test_uncorr_confusion_bva.png'))

        visualize_predictions_separate(train_loader, os.path.join(save_dir, 'prediction_vis_separate', 'train_bva'), M=64)
        visualize_predictions_separate(val_loader, os.path.join(save_dir, 'prediction_vis_separate', 'val_bva'), M=64)
        visualize_predictions_separate(test_loader_anticorrelated, os.path.join(save_dir, 'prediction_vis_separate', 'test_anticorr_bva'), M=64)
        visualize_predictions_separate(test_loader_uncorrelated, os.path.join(save_dir, 'prediction_vis_separate', 'test_uncorr_bva'), M=64)

        save_models(prefix='bva')


    epoch_dict['epoch'].append(epoch)
    epoch_dict['trn_loss'].append(trn_loss)
    epoch_dict['val_loss'].append(val_loss)
    epoch_dict['tst_ac_loss'].append(tst_ac_loss)
    epoch_dict['tst_uc_loss'].append(tst_uc_loss)
    epoch_dict['trn_f1_acc'].append(trn_f1_acc)
    epoch_dict['trn_f2_acc'].append(trn_f2_acc)
    epoch_dict['trn_acc'].append((trn_f1_acc + trn_f2_acc) / 2.0)
    epoch_dict['val_f1_acc'].append(val_f1_acc)
    epoch_dict['val_f2_acc'].append(val_f2_acc)
    epoch_dict['val_acc'].append((val_f1_acc + val_f2_acc) / 2.0)
    epoch_dict['tst_ac_f1_acc'].append(tst_ac_f1_acc)
    epoch_dict['tst_uc_f1_acc'].append(tst_uc_f1_acc)
    epoch_dict['tst_ac_f2_acc'].append(tst_ac_f2_acc)
    epoch_dict['tst_uc_f2_acc'].append(tst_uc_f2_acc)
    epoch_dict['tst_ac_acc'].append((tst_ac_f1_acc + tst_ac_f2_acc) / 2.0)
    epoch_dict['tst_uc_acc'].append((tst_uc_f1_acc + tst_uc_f2_acc) / 2.0)
    epoch_dict['trn_l_acc_dict'].append(trn_l_acc_dict)
    epoch_dict['trn_r_acc_dict'].append(trn_r_acc_dict)
    epoch_dict['val_l_acc_dict'].append(val_l_acc_dict)
    epoch_dict['val_r_acc_dict'].append(val_r_acc_dict)
    epoch_dict['tst_ac_l_acc_dict'].append(tst_ac_l_acc_dict)
    epoch_dict['tst_ac_r_acc_dict'].append(tst_ac_r_acc_dict)

    fig = plt.figure(figsize=(6,4))
    plt.plot(epoch_dict['epoch'], epoch_dict['trn_loss'], linewidth=2, label='Train')
    plt.plot(epoch_dict['epoch'], epoch_dict['val_loss'], linewidth=2, label='Val')
    # plt.plot(epoch_dict['epoch'], epoch_dict['tst_loss'], linewidth=2, label='Test')
    plt.plot(epoch_dict['epoch'], epoch_dict['tst_ac_loss'], linewidth=2, label='Test AC')
    plt.plot(epoch_dict['epoch'], epoch_dict['tst_uc_loss'], linewidth=2, label='Test UC')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(fontsize=18, fancybox=True, framealpha=0.3)
    sns.despine()
    plt.savefig(os.path.join(save_dir, 'epoch_losses.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, 'epoch_losses.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig = plt.figure(figsize=(6,4))
    plt.plot(epoch_dict['epoch'], epoch_dict['trn_acc'], linewidth=2, label='Train')
    plt.plot(epoch_dict['epoch'], epoch_dict['val_acc'], linewidth=2, label='Val')
    plt.plot(epoch_dict['epoch'], epoch_dict['tst_ac_acc'], linewidth=2, label='Test AC')
    plt.plot(epoch_dict['epoch'], epoch_dict['tst_uc_acc'], linewidth=2, label='Test UC')
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Accuracy'.format(args.target_variable1), fontsize=18)
    plt.legend(fontsize=18, fancybox=True, framealpha=0.3)
    sns.despine()
    plt.savefig(os.path.join(save_dir, 'epoch_acc.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, 'epoch_acc.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig = plt.figure(figsize=(6,4))
    plt.plot(epoch_dict['epoch'], epoch_dict['trn_f1_acc'], linewidth=2, label='Train')
    plt.plot(epoch_dict['epoch'], epoch_dict['val_f1_acc'], linewidth=2, label='Val')
    # plt.plot(epoch_dict['epoch'], epoch_dict['tst_f1_acc'], linewidth=2, label='Test')
    plt.plot(epoch_dict['epoch'], epoch_dict['tst_ac_f1_acc'], linewidth=2, label='Test AC')
    plt.plot(epoch_dict['epoch'], epoch_dict['tst_uc_f1_acc'], linewidth=2, label='Test UC')
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Accuracy'.format(args.target_variable1), fontsize=18)
    plt.legend(fontsize=18, fancybox=True, framealpha=0.3)
    sns.despine()
    plt.savefig(os.path.join(save_dir, 'epoch_f1_acc.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, 'epoch_f1_acc.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig = plt.figure(figsize=(6,4))
    plt.plot(epoch_dict['epoch'], epoch_dict['trn_f2_acc'], linewidth=2, label='Train')
    plt.plot(epoch_dict['epoch'], epoch_dict['val_f2_acc'], linewidth=2, label='Val')
    # plt.plot(epoch_dict['epoch'], epoch_dict['tst_f2_acc'], linewidth=2, label='Test')
    plt.plot(epoch_dict['epoch'], epoch_dict['tst_ac_f2_acc'], linewidth=2, label='Test AC')
    plt.plot(epoch_dict['epoch'], epoch_dict['tst_uc_f2_acc'], linewidth=2, label='Test UC')
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Accuracy'.format(args.target_variable2), fontsize=18)
    plt.legend(fontsize=18, fancybox=True, framealpha=0.3)
    sns.despine()
    plt.savefig(os.path.join(save_dir, 'epoch_f2_acc.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, 'epoch_f2_acc.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    if args.z_dim == 2:  # If the latent space is 2D, plot it
        fig = plt.figure(figsize=(6,4))

        for combo in itertools.product(possible_labels, possible_labels):
            indexes = np.all(trn_labels == np.array(combo), axis=1)
            z_subset = trn_zs[indexes,:]
            plt.scatter(z_subset[:,0], z_subset[:,1], alpha=0.2, label='$z_1$={}, $z_2$={}'.format(combo[0], combo[1]))
        # plt.scatter(trn_zs[:,0], trn_zs[:,1], alpha=0.2)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('$z_1$', fontsize=18)
        plt.ylabel('$z_2$', fontsize=18)
        plt.legend(fontsize=16, fancybox=True, framealpha=0.3)
        sns.despine()
        plt.savefig(os.path.join(save_dir, 'z_space', 'epoch_train_z_space_{}.png'.format(epoch)), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        fig = plt.figure(figsize=(6,4))
        for combo in itertools.product(possible_labels, possible_labels):
            indexes = np.all(tst_labels == np.array(combo), axis=1)
            z_subset = tst_zs[indexes,:]
            plt.scatter(z_subset[:,0], z_subset[:,1], alpha=0.2, label='$z_1$={}, $z_2$={}'.format(combo[0], combo[1]))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('$z_1$', fontsize=18)
        plt.ylabel('$z_2$', fontsize=18)
        plt.legend(fontsize=16, fancybox=True, framealpha=0.3)
        sns.despine()
        plt.savefig(os.path.join(save_dir, 'z_space', 'epoch_test_z_space_{}.png'.format(epoch)), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print('Epoch: {:3d} | Trn loss: {:6.4e} | Trn f1: {:6.4f} | Trn f2: {:6.4f} | '
          'Val loss: {:6.4e} | Val f1: {:6.4f} | Val f2: {:6.4f} | '
          'Tst loss: {:6.4e} | Tst f1: {:6.4f} | Tst f2: {:6.4f}'.format(
           epoch, trn_loss, trn_f1_acc, trn_f2_acc, val_loss, val_f1_acc, val_f2_acc, tst_ac_loss, tst_ac_f1_acc, tst_ac_f2_acc))
    sys.stdout.flush()

    stats_dict = { 'epoch': epoch,
                   'global_iteration': global_iteration,
                   'trn_loss': trn_loss,
                   'trn_f1_acc': trn_f1_acc,
                   'trn_f2_acc': trn_f2_acc,
                   'val_loss': val_loss,
                   'val_f1_acc': val_f1_acc,
                   'val_f2_acc': val_f2_acc,
                   'tst_ac_loss': tst_ac_loss,
                   'tst_uc_loss': tst_uc_loss,
                   'tst_ac_f1_acc': tst_ac_f1_acc,
                   'tst_uc_f1_acc': tst_uc_f1_acc,
                   'tst_ac_f2_acc': tst_ac_f2_acc,
                   'tst_uc_f2_acc': tst_uc_f2_acc,
                 }

    # Save stats to CSV file and Tensorboard
    # ---------------------------------------
    iteration_logger.writerow(stats_dict)
    for name in stats_dict:
        writer.add_scalar(name, stats_dict[name], global_iteration)
    # ---------------------------------------


    # Now train the model
    # ---------------------------------------
    for images, labels in tqdm(train_loader):
        images = images.to(use_device)
        f1_labels = labels[:,0]
        f2_labels = labels[:,1]

        f1_labels = f1_labels.to(use_device)
        f2_labels = f2_labels.to(use_device)

        z = model(images)
        z1 = z[:,:subspace_dim]
        z2 = z[:,subspace_dim:]

        f1_logits = f1_classifier(z1)
        f2_logits = f2_classifier(z2)

        f1_xentropy_loss = F.cross_entropy(f1_logits, f1_labels)
        f2_xentropy_loss = F.cross_entropy(f2_logits, f2_labels)

        G_loss = (f1_xentropy_loss + f2_xentropy_loss) / 2.0

        G_gan_loss = 0
        if args.mi_type == 'unconditional':
            perm = torch.randperm(len(z))
            z_shuffled = torch.clone(z)
            z_shuffled[:,:subspace_dim] = z_shuffled[perm,:subspace_dim]
            real_scores_latent = discriminator(z)
            fake_scores_latent = discriminator(z_shuffled)
            G_gan_loss = 0.0
            G_gan_loss += F.binary_cross_entropy_with_logits(fake_scores_latent, torch.ones((z.size(0), 1), device=z.device))
            G_gan_loss += F.binary_cross_entropy_with_logits(real_scores_latent, torch.zeros((z.size(0), 1), device=z.device))
            G_loss += args.disentangle_weight * G_gan_loss
        elif args.mi_type == 'conditional':
            # restricted_sampling(z, f1_labels, f2_labels, dim, label)

            # for f1_value in list(set(f1_labels.detach().cpu().numpy())):  # Possible values for a1
            #     value_filter = (f1_labels == f1_value)
            #     cond_zs = z[value_filter]  # Take only the zs that have the value of a1 that we're conditioning on
            #     cond_zs_sub = cond_zs
            #     perm = torch.randperm(len(cond_zs_sub))
            #     cond_zs_sub_shuffled = torch.clone(cond_zs_sub)
            #     cond_zs_sub_shuffled[:,:subspace_dim] = cond_zs_sub_shuffled[perm,:subspace_dim]
            #     # mmd_loss_latent += mmd_loss(cond_zs_sub, cond_zs_sub_shuffled, sigmas=sigmas)
            #     # cond_left_labels = f1_labels[value_filter].unsqueeze(1)
            #     cond_left_labels = F.one_hot(f1_labels[value_filter], num_f1_values)
            #     real_scores_f1 = f1_discriminator(torch.cat([cond_zs, cond_left_labels], dim=1))
            #     fake_scores_f1 = f1_discriminator(torch.cat([cond_zs_sub_shuffled, cond_left_labels], dim=1))
            #     G_gan_loss_f1 = F.binary_cross_entropy_with_logits(fake_scores_f1, torch.ones((cond_zs.size(0), 1), device=z.device)) + \
            #                     F.binary_cross_entropy_with_logits(real_scores_f1, torch.zeros((cond_zs.size(0), 1), device=z.device))  # Original G objective
            #     G_gan_loss += G_gan_loss_f1

            # for f2_value in list(set(f2_labels.detach().cpu().numpy())):
            #     value_filter = (f2_labels == f2_value)
            #     cond_zs = z[value_filter]  # Take only the zs that have the value of a1 that we're conditioning on
            #     cond_zs_sub = cond_zs
            #     perm = torch.randperm(len(cond_zs_sub))
            #     cond_zs_sub_shuffled = torch.clone(cond_zs_sub)
            #     cond_zs_sub_shuffled[:,subspace_dim:] = cond_zs_sub_shuffled[perm,subspace_dim:]
            #     # mmd_loss_latent += mmd_loss(cond_zs_sub, cond_zs_sub_shuffled, sigmas=sigmas)
            #     # cond_left_labels = f1_labels[value_filter].unsqueeze(1)
            #     cond_right_labels = F.one_hot(f2_labels[value_filter], num_f2_values)
            #     real_scores_f2 = f2_discriminator(torch.cat([cond_zs, cond_right_labels], dim=1))
            #     fake_scores_f2 = f2_discriminator(torch.cat([cond_zs_sub_shuffled, cond_right_labels], dim=1))
            #     G_gan_loss_f2 = F.binary_cross_entropy_with_logits(fake_scores_f2, torch.ones((cond_zs.size(0), 1), device=z.device)) + \
            #                     F.binary_cross_entropy_with_logits(real_scores_f2, torch.zeros((cond_zs.size(0), 1), device=z.device))  # Original G objective
            #     G_gan_loss += G_gan_loss_f2

            G_gan_loss = 0
            for discr in discr_list:
                idb.set_trace()
                loss_discriminator = discr.train(v=z.detach())
                loss_disentanglement += discr.eval_reals(v=z)
                loss_disentanglement += discr.eval_fakes(v=z)

            loss_disentanglement /= dim * 2 * 2 # *two condition_values *reals/fakes
            loss += weight_disentanglement * loss_disentanglement

            G_loss += args.disentangle_weight * G_gan_loss

        optimizer.zero_grad()
        G_loss.backward()
        optimizer.step()

        # Train linear classification heads
        # ---------------------------------
        for cls_train_iter in range(args.num_cls_steps):
            f1_logits = f1_classifier(z1.detach())
            f2_logits = f2_classifier(z2.detach())
            f1_xentropy_loss = F.cross_entropy(f1_logits, f1_labels)
            f2_xentropy_loss = F.cross_entropy(f2_logits, f2_labels)
            classification_loss = (f1_xentropy_loss + f2_xentropy_loss) / 2.0
            classification_optimizer.zero_grad()
            classification_loss.backward()
            classification_optimizer.step()
        # ---------------------------------

        # if args.mi_type == 'unconditional':
        #     real_scores_latent = discriminator(z.detach())
        #     fake_scores_latent = discriminator(z_shuffled.detach())
        #     D_real_loss_latent = F.binary_cross_entropy_with_logits(real_scores_latent, torch.ones((z.size(0), 1), device=z.device))
        #     D_fake_loss_latent = F.binary_cross_entropy_with_logits(fake_scores_latent, torch.zeros((z.size(0), 1), device=z.device))
        #     D_loss = D_real_loss_latent + D_fake_loss_latent

        #     disc_optimizer.zero_grad()
        #     D_loss.backward()
        #     disc_optimizer.step()
        # elif args.mi_type == 'conditional':
        #     D_loss = 0.0

        #     for f1_value in list(set(f1_labels.detach().cpu().numpy())):  # Possible values for a1
        #         value_filter = (f1_labels == f1_value)
        #         cond_zs = z[value_filter]  # Take only the zs that have the value of a1 that we're conditioning on
        #         cond_zs_sub = cond_zs
        #         perm = torch.randperm(len(cond_zs_sub))
        #         cond_zs_sub_shuffled = torch.clone(cond_zs_sub)
        #         cond_zs_sub_shuffled[:,:subspace_dim] = cond_zs_sub_shuffled[perm,:subspace_dim]
        #         cond_left_labels = F.one_hot(f1_labels[value_filter], num_f1_values)
        #         real_scores_f1 = f1_discriminator(torch.cat([cond_zs.detach(), cond_left_labels], dim=1))
        #         fake_scores_f1 = f1_discriminator(torch.cat([cond_zs_sub_shuffled.detach(), cond_left_labels], dim=1))
        #         D_gan_loss_fg = F.binary_cross_entropy_with_logits(real_scores_f1, torch.ones((cond_zs.size(0), 1), device=z.device)) + \
        #                         F.binary_cross_entropy_with_logits(fake_scores_f1, torch.zeros((cond_zs.size(0), 1), device=z.device))  # Original G objective
        #         D_loss += D_gan_loss_fg

        #     for f2_value in list(set(f2_labels.detach().cpu().numpy())):
        #         value_filter = (f2_labels == f2_value)
        #         cond_zs = z[value_filter]  # Take only the zs that have the value of a1 that we're conditioning on
        #         cond_zs_sub = cond_zs
        #         perm = torch.randperm(len(cond_zs_sub))
        #         cond_zs_sub_shuffled = torch.clone(cond_zs_sub)
        #         cond_zs_sub_shuffled[:,subspace_dim:] = cond_zs_sub_shuffled[perm,subspace_dim:]
        #         cond_right_labels = F.one_hot(f2_labels[value_filter], num_f2_values)
        #         real_scores_f2 = f2_discriminator(torch.cat([cond_zs.detach(), cond_right_labels], dim=1))
        #         fake_scores_f2 = f2_discriminator(torch.cat([cond_zs_sub_shuffled.detach(), cond_right_labels], dim=1))
        #         D_gan_loss_bg = F.binary_cross_entropy_with_logits(real_scores_f2, torch.ones((cond_zs.size(0), 1), device=z.device)) + \
        #                         F.binary_cross_entropy_with_logits(fake_scores_f2, torch.zeros((cond_zs.size(0), 1), device=z.device))  # Original G objective
        #         D_loss += D_gan_loss_bg

        #     disc_optimizer.zero_grad()
        #     D_loss.backward()
        #     disc_optimizer.step()


        if args.plot_covariance and (global_iteration % 1000 == 0):
            z_np = z.detach().cpu().numpy()
            z_np -= z_np.mean(axis=0)  # Subtract mean
            z_np /= z_np.std(axis=0)   # Standardize
            covariance = (np.dot(z_np.T, z_np) / (z_np.shape[0]-1))  # Compute covariance

            fig = plt.figure()
            plt.imshow(covariance, cmap='seismic', vmin=-1, vmax=1)
            plt.colorbar()
            plt.savefig(os.path.join(save_dir, 'cov_plots', 'cov_{}.png'.format(global_iteration)), bbox_inches='tight')
            plt.close(fig)

        global_iteration += 1

        iteration_dict['iteration'].append(global_iteration)
        iteration_dict['f1_xentropy_loss'].append(f1_xentropy_loss)
        iteration_dict['f2_xentropy_loss'].append(f2_xentropy_loss)
        iteration_dict['G_loss'].append(G_loss)

        if args.mi_type in ['unconditional', 'conditional']:
            iteration_dict['G_gan_loss'].append(G_gan_loss)
            iteration_dict['D_loss'].append(D_loss)

        if global_iteration % args.plot_every == 0:
            for key in [mykey for mykey in iteration_dict.keys() if mykey != 'iteration']:
                fig = plt.figure(figsize=(6,4))
                plt.plot(iteration_dict['iteration'], iteration_dict[key], linewidth=3)
                plt.xlabel('Iteration', fontsize=18)
                plt.ylabel(key, fontsize=18)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.title(key, fontsize=20)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'loss_plots', '{}.png'.format(key)), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

        if global_iteration % args.eval_mi_every == 0:
            mi_dict['iteration'].append(global_iteration)

            z1_np = z1.detach().cpu().numpy()
            z2_np = z2.detach().cpu().numpy()
            f1_labels_np = f1_labels.unsqueeze(1).detach().cpu().float().numpy()
            f2_labels_np = f2_labels.unsqueeze(1).detach().cpu().float().numpy()

            cmi_z1_z2_given_f1 = continuous.get_pmi(z1_np,
                                                    z2_np,
                                                    f1_labels_np,
                                                    k=3,
                                                    norm='max',
                                                    estimator='fp')
            cmi_z1_z2_given_f2 = continuous.get_pmi(z1_np,
                                                    z2_np,
                                                    f2_labels_np,
                                                    k=3,
                                                    norm='max',
                                                    estimator='fp')
            cmi_z1_f2_given_f1 = continuous.get_pmi(z1_np,
                                                    f2_labels_np,
                                                    f1_labels_np,
                                                    k=3,
                                                    norm='max',
                                                    estimator='fp')
            cmi_z2_f1_given_f2 = continuous.get_pmi(z2_np,
                                                    f1_labels_np,
                                                    f2_labels_np,
                                                    k=3,
                                                    norm='max',
                                                    estimator='fp')

            mi_dict['cmi_z1_z2_given_f1'].append(cmi_z1_z2_given_f1)
            mi_dict['cmi_z1_z2_given_f2'].append(cmi_z1_z2_given_f2)
            mi_dict['cmi_z1_f2_given_f1'].append(cmi_z1_f2_given_f1)
            mi_dict['cmi_z2_f1_given_f2'].append(cmi_z2_f1_given_f2)
            mi_dict['z1_z2_mi'].append(continuous.get_mi(z1_np, z2_np, estimator='ksg', k=args.k_neighbors))
            mi_dict['z1_f1_mi'].append(continuous.get_mi(z1_np, f1_labels.detach().cpu().numpy(), estimator='ksg', k=args.k_neighbors))
            mi_dict['z1_f2_mi'].append(continuous.get_mi(z1_np, f2_labels.detach().cpu().numpy(), estimator='ksg', k=args.k_neighbors))
            mi_dict['z2_f1_mi'].append(continuous.get_mi(z2_np, f1_labels.detach().cpu().numpy(), estimator='ksg', k=args.k_neighbors))
            mi_dict['z2_f2_mi'].append(continuous.get_mi(z2_np, f2_labels.detach().cpu().numpy(), estimator='ksg', k=args.k_neighbors))
            # mi_dict['f1_f2_mi'].append(continuous.get_mi(f1_labels.detach().cpu().numpy(), f2_labels.detach().cpu().numpy(), estimator='ksg', k=args.k_neighbors))

        if global_iteration % args.plot_every == 0:
            for key in [mykey for mykey in mi_dict.keys() if mykey != 'iteration']:
                fig = plt.figure(figsize=(6,4))
                plt.plot(mi_dict['iteration'], mi_dict[key], linewidth=3)
                plt.xlabel('Iteration', fontsize=18)
                plt.ylabel(key, fontsize=18)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.title(key, fontsize=20)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'mi_plots', '{}.png'.format(key)), bbox_inches='tight', pad_inches=0)
                plt.close(fig)
    # ---------------------------------------


# Save the final trained model and classifiers
save_models()

# Save all epoch_dict stats accumulated over the course of training
with open(os.path.join(save_dir, 'result.pkl'), 'wb') as f:
    pkl.dump(epoch_dict, f)
