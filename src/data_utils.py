from torchvision.datasets import CIFAR100, SUN397
from torchvision.models import resnet18, ResNet18_Weights
import torchvision
from torchvision import transforms
from typing import Callable, Optional, List
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import torch
from src.simple_utils import load_pickle
import pathlib
import json
import os
import logging
from collections import defaultdict, Counter
import torchvision.transforms.functional as TF
import random

from src.datasets.amazon_reviews_utils import *
import torch.multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')

# log = logging.getLogger(__name__)
log = logging.getLogger("app")

osj = os.path.join

class SUN397Dataset(SUN397):
    def __init__(self, root, features=None, transform=None):
        super(SUN397Dataset, self).__init__(root, transform=transform)
        self.class_to_superclass = {}
        self.features = features
        self.targets = self._labels
        self.supertarget_transform = None
        if self.features is not None:
            self.input_features = torch.load(features)
            if 'supertargets' in self.input_features.keys():
                self.input_features.pop('supertargets')
            for key,value in self.input_features.items():
                self.input_features[key] = value.cpu().detach()
            
    def __getitem__(self,index):
        if self.features is not None:
            sample, target, feat_index = self.input_features['features'][index], int(self.input_features['targets'][index].item()), int(self.input_features['indices'][index].item())
            assert feat_index==index
        else:
            sample, target = super(SUN397Dataset, self).__getitem__(index)
        if len(self.class_to_superclass)>0:
            supertarget = self.class_to_superclass[target]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.supertarget_transform is not None:
            supertarget = self.super_target_transform(supertarget)
        return sample, supertarget, index # target # return index for trainKPU otherwise return target.

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = np.array(targets).astype(np.int_)

        self.target_transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class DatasetwithSentiments(torch.utils.data.Dataset):
    def __init__(self, data, targets, sentiments, arch):
        self.data = data
        self.targets = np.array(targets).astype(np.int_)
        self.sentiments = np.array(sentiments).astype(np.int_)
        self.arch = arch
        self.target_transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.arch=='Roberta:':
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
            inputs = tokenizer.encode_plus(self.data[idx], None, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_token_type_ids=True)
            data = {'input_ids':torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask':torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids':torch.tensor(inputs['token_type_ids'], dtype=torch.long)
            }
            return self.data, self.targets[idx], self.sentiments[idx]

        return self.data[idx], self.targets[idx], self.sentiments[idx]


# class AmazonReviewsRobertaFeatures(torch.utils.data.Dataset):
#     def __init__(self, data_dir, targets, sentiments, arch):

def get_labels(targets): 
    counter = Counter(targets)
    return sorted(list(counter.keys()))

def get_size_per_class(dataset):

    if isinstance(dataset, Subset):
        targets = np.array(dataset.dataset.targets)[dataset.indices] 
        counter = Counter(targets)
    else: 
        counter = Counter(dataset.targets)

    return counter


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    
    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        transform_idx = self.transform_idx
        return (data[0], data[1], transform_idx[index]) + data[2:]

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def dataset_transform_labels(cls): 

    def __getitem__(self, index):
        
        data = cls.__getitem__(self, index)
        
        return (data[0], self.target_transform(data[1])) + data[2:]
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def get_data(data_dir, dataset, train = None, transform=None):

    if dataset.lower() == "cifar100":
        CIFAR100withIndices = dataset_with_indices(CIFAR100)
        data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=train, transform=transform, download=True)

        return data

    else:
        raise NotImplementedError("Please add support for %s dataset" % dataset)
    

def get_combined_data(data_dir, dataset, arch, ood_class=None, transform=None, train_fraction = None , seed=42, mode='domain_disc', use_superclass=False):
    np.random.seed(seed)

    if dataset.lower() == "cifar100":
        CIFAR100withIndices = dataset_with_indices(CIFAR100)
        train_data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=True, transform=transform[0], download=True)
        val_data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=False, transform=transform[1], download=True)
        train_data_AL = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=True, transform=transform[1], download=True)

        return train_data, val_data, train_data_AL

    elif dataset.lower() == "sun397":
        train_txt = pd.read_csv(f"{data_dir}/sun397/Partitions/Training_01.txt", header=None)
        test_txt = pd.read_csv(f"{data_dir}/sun397/Partitions/Testing_01.txt", header=None)
        train_files = [train_txt.iloc[i,0] for i in range(len(train_txt))]
        test_files = [test_txt.iloc[i,0] for i in range(len(test_txt))]

        if arch=='Resnet50':
            data = SUN397Dataset(root=f"{data_dir}/sun397", features=f"{data_dir}/train_"+dataset+"_ResNet50_features_imagenet_pretraining.pth")
        elif arch=='CLIP_RN50':
            data = SUN397Dataset(root=f"{data_dir}/sun397", features=f"{data_dir}/train_"+dataset+"_CLIP_RN50_features_pretrained.pth")
        elif arch=='CLIP_ViT-L14':
            data = SUN397Dataset(root=f"{data_dir}/sun397", features=f"{data_dir}/train_"+dataset+"_CLIP_ViT-L14_features_pretrained.pth")
        file_names = [i.__str__().split('SUN397')[-1] for i in data._image_files]
        train_idxs = [i for i in range(len(file_names)) if file_names[i] in train_files]
        val_idxs = [i for i in range(len(file_names)) if file_names[i] in test_files]

        # ResNet50 pretrained on ImageNet features over SUN397
        train_data = Subset(data, train_idxs)
        val_data = Subset(data, val_idxs)

        return train_data, val_data

    elif dataset.lower() == "amazon_reviews":

        if arch=="Roberta_linear_classifier":
            data, targets, sentiments, _ = get_amazon_reviews_features(f"{data_dir}/amazon_reviews_roberta_features.pth", ood_class, arch)
        else:
            data, targets, sentiments, _ = get_amazon_reviews(f"{data_dir}/amazon_reviews_tp", ood_class, arch)
        labels = get_labels(targets)
        sentiment_labels = get_labels(sentiments)

        idx_per_class, sent_idx = [],[]

        for label in labels:
            for sent_label in sentiment_labels:
                idx_i = np.intersect1d(np.where(targets == label)[0], np.where(sentiments == sent_label)[0])
                np.random.shuffle(idx_i)
                idx_per_class.append(idx_i)

        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(idx_per_class))])
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(idx_per_class))])

        train_data = DatasetwithSentiments(data[train_idx], targets[train_idx], sentiments[train_idx], arch)
        val_data = DatasetwithSentiments(data[val_idx], targets[val_idx], sentiments[val_idx], arch)
        train_data_AL = DatasetwithSentiments(data[train_idx], targets[train_idx], sentiments[train_idx], arch)

        return train_data, val_data, train_data_AL

    else:
        raise NotImplementedError("Please add support for %s dataset" % dataset)


def get_classes(classes : List): 
    if isinstance(classes[0], list):
        return [list(map(int, i)) for i in classes]
    else:  
        return list(map(int, classes))

def get_marginal(marginal_type: str, marginal:  List[int], num_classes: int): 
    if marginal_type == "Uniform": 
        return np.array([1.0/num_classes]*num_classes)
    elif marginal_type == "Dirichlet": 
        return np.random.dirichlet(marginal[0]*num_classes)
    elif marginal_type == "Manual":
        marginal =  np.array(marginal)
        assert np.sum(marginal) == 1.0
        return marginal
    else: 
        raise NotImplementedError("Please check your marginal type for source and target")


def get_idx(targets, classes, total_per_class):

    idx = None
    log.debug(f"Target length {len(targets)} of type {type(targets)} and elements are {targets[:50]}...")
    targets = np.array(targets)
    for i in range(len(classes)):
        c_idx = None
        if isinstance(classes[i], list): 
            log.debug(f"Class {i} is a list {classes[i]}")
            for j in classes[i]:
                log.debug(f"Class {i} has {type(j)} {j}")
                if c_idx is None: 
                    c_idx = np.where(j == targets)[0]
                else: 
                    c_idx = np.concatenate((c_idx, np.where(j == targets)[0]))
            log.debug(f"Number of instances for class {i} are {len(c_idx)}")
        else: 
            log.debug(f"Class {i} is a {type(classes[i])} {classes[i]}")
            c_idx = np.where(classes[i] == targets)[0]
            log.debug(f"Number of instances for class {i} are {len(c_idx)}")

        if len(c_idx) >= total_per_class[i]:     
            c_idx = np.random.choice(c_idx, size = total_per_class[i], replace= False)
        else: 
            log.error("Not enough samples to get the split for class %d. \n\
                       Needed %f. Obtained %f" %(i, total_per_class[i], len(c_idx)))
        
        if idx is None:
            idx = [c_idx]
        else: 
            idx.append(c_idx) 

    label_map = {}
    for i in range(len(classes)): 
        if isinstance(classes[i], list): 
            for j in classes[i]:
                label_map[j] = i
        else: 
            label_map[classes[i]] = i
    
    log.debug(label_map)
    target_transform = lambda x: label_map[x]
    
    return idx, target_transform


def split_indicies(targets, source_classes, target_classes,\
     source_marginal, target_marginal, source_size, target_size): 

    source_per_class = np.concatenate((np.array([ int(i*source_size) for i in source_marginal]),\
         np.array([0], dtype=np.int32)))
    target_per_class = np.array([ int(i*target_size) for i in target_marginal])

    total_per_class = source_per_class + target_per_class

    log.debug(f"Needed <{source_per_class}> samples for source")
    log.debug(f"Needed <{target_per_class}> samples for target")
    
    idx, target_transform = get_idx(targets, target_classes, total_per_class)
    
    source_idx = [idx[c][:source_per_class[c]] for c in range(len(source_classes))]
    target_idx = [idx[c][source_per_class[c]:] for c in range(len(target_classes))]

    return source_idx, target_idx, target_transform


def split_indicies_with_size(targets, source_classes, target_classes,
                             source_marginal, target_marginal, size_per_class):

    source_per_class = np.concatenate((np.array([ int(source_marginal[class_idx]*size_per_class[i]) for  class_idx, i in enumerate(source_classes)]),\
            np.array([0], dtype=np.int32)))

    target_per_class = np.array([ int(target_marginal[class_idx]*size_per_class[i]) for  class_idx, i in enumerate(target_classes[:-1])])
    
    len_ood_data = np.sum([size_per_class[i] for i in target_classes[-1]])

    target_per_class = np.concatenate((target_per_class, np.array([len_ood_data], dtype=np.int32)))

    total_per_class = source_per_class + target_per_class
    
    log.debug(f"Needed <{source_per_class}> samples for source")
    log.debug(f"Needed <{target_per_class}> samples for target")
    
    idx, target_transform = get_idx(targets, target_classes, total_per_class)
    
    source_idx = [idx[c][:source_per_class[c]] for c in range(len(source_classes))]
    target_idx = [idx[c][source_per_class[c]:] for c in range(len(target_classes))]
    # import pdb; pdb.set_trace()
    return source_idx, target_idx, target_transform

def remap_idx(idx): 
    default_func = lambda: -1 

    def_map = defaultdict(default_func)
    sorted_idx = np.sort(idx)

    for i in range(len(sorted_idx)):
        def_map[sorted_idx[i]] = i
    
    return def_map


def get_splits(data_dir, dataset, source_classes, source_marginal, source_size,\
    target_classes, target_marginal, target_size, train = False, transform: Optional[Callable] = None): 

    data = get_data(data_dir, dataset, train=train, transform=transform)

    source_idx, target_idx, target_transform = split_indicies(data.targets, source_classes, target_classes,\
        source_marginal, target_marginal, source_size, target_size)
    
    data.target_transform = target_transform

    data.transform_idx = remap_idx(np.concatenate(target_idx).ravel())

    source_per_class = []
    for i in range(len(source_idx)):
        source_per_class.append(Subset(data, source_idx[i]))

    log.debug("Creating labeled and unlabeled splits}")
    source_data = Subset(data, np.concatenate(source_idx).ravel())
    target_data = Subset(data, np.concatenate(target_idx).ravel())

    return source_per_class, source_data, target_data


def get_splits_from_data(data, source_classes, source_marginal,\
    target_classes, target_marginal, dataset='', train = False, transform: Optional[Callable] = None):
    
    size_per_class = get_size_per_class(data)

    log.debug(f"Size per class: {size_per_class}")
    
    if isinstance(data, Subset):
        targets = np.array(data.dataset.targets)[data.indices] 
        source_idx, target_idx, target_transform = split_indicies_with_size(targets, source_classes, target_classes,\
            source_marginal, target_marginal, size_per_class)
        source_idx = [[data.indices[j]] for i in source_idx for j in i]
        target_idx = [[data.indices[j]] for i in target_idx for j in i] 
    else:
        source_idx, target_idx, target_transform = split_indicies_with_size(data.targets, source_classes, target_classes,\
            source_marginal, target_marginal, size_per_class)
    
    source_idx = np.concatenate(source_idx).ravel()
    target_idx = np.concatenate(target_idx).ravel()

    if isinstance(data, Subset):
        data.dataset.transform_idx = remap_idx(target_idx)
    else:
        data.transform_idx = remap_idx(target_idx)

    log.debug("Creating labeled and unlabeled splits}")

    SubsetwithTransform = dataset_transform_labels(Subset)
    
    if isinstance(data, Subset):
        source_data = SubsetwithTransform(data.dataset, source_idx) if dataset != 'sun397' else Subset(data.dataset, source_idx)
        target_data = SubsetwithTransform(data.dataset, target_idx) if dataset != 'sun397' else Subset(data.dataset, target_idx)
        
    else:
        source_data = SubsetwithTransform(data, source_idx) if dataset != 'sun397' else Subset(data, source_idx)
        target_data = SubsetwithTransform(data, target_idx) if dataset != 'sun397' else Subset(data, target_idx)
    source_data.target_transform = target_transform  
    target_data.target_transform = target_transform
    # import pdb; pdb.set_trace()
    return source_data, target_data, source_idx, target_idx, target_transform



def get_preprocessing(dset: str, use_aug: bool = False, train: bool = False, mean=None, std=None, arch=None):

    log.info(f"Using {dset} dataset with augmentation {use_aug} and training {train}")
    if dset.lower() == 'cifar100':
        mean = (0.5074, 0.4867, 0.4411) if mean is None else mean
        std = (0.2011, 0.1987, 0.2025) if std is None else std
    else:
        mean = (0.5, 0.5, 0.5) if mean is None else mean
        std = (0.5, 0.5, 0.5) if std is None else std

    if dset.lower().startswith("cifar"):
        if use_aug and train:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )
    else:
        transform = None

    return transform
