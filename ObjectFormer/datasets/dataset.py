import os

import cv2
import torch
from torch.utils.data import Dataset

from ObjectFormer.datasets.utils import get_transformations
from ObjectFormer.utils.registries import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class TamperingDataset(Dataset):
    def __init__(self, mode, dataset_cfg):
        assert mode in [
            'train',
            'val',
            'test',
        ], 'mode should be train/val/test'
        # self.dataset_name = dataset_cfg['DATASET_NAME']
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.root_dir = dataset_cfg['ROOT_DIR']
        
        print("*"*100)
        print(dataset_cfg[mode.upper()+"_SPLIT"])
        if isinstance(dataset_cfg[mode.upper()+"_SPLIT"], str):
            split_file = os.path.join(self.root_dir, dataset_cfg[mode.upper()+"_SPLIT"])
            with open(split_file) as f:
                self.info_list = f.readlines()
        
        elif isinstance(dataset_cfg[mode.upper()+"_SPLIT"], list):
            self.info_list = []
            for split in dataset_cfg[mode.upper()+"_SPLIT"]:
                split_file = os.path.join(self.root_dir, split)
                with open(split_file) as f:
                    self.info_list.extend(f.readlines())
        
        self.augmentations = get_transformations(mode, dataset_cfg)
        self.return_mask = dataset_cfg['RETURN_MASK']
        image_info = self.info_list[0].strip().split()
        if self.return_mask:
            assert len(image_info) > 2, f'return_mask set to True, but no mask path found.'

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        info_line = self.info_list[idx]
        image_info = info_line.strip().split('\t')
        
        image_path, label = image_info[0], int(image_info[2]!='0')
        image_full_path = os.path.join(self.root_dir, image_path)
        
        img = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.return_mask:
            img = self.augmentations(image=img)['image']
            mask = None
        elif label == 0:
            img = self.augmentations(image=img)['image']
            _, H, W = img.size()
            mask = torch.zeros((H, W))
        else:
            mask_full_path = os.path.join(self.root_dir, image_info[1])
            mask = cv2.imread(mask_full_path,cv2.IMREAD_GRAYSCALE)
            augmented = self.augmentations(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        
        mask=mask/255
        sample = {}
        sample['img'] = torch.FloatTensor(img.float())
        sample['mask'] = torch.LongTensor(mask.long()) if mask is not None else None
        sample['mask_2ch'] = torch.FloatTensor(torch.cat([(mask.unsqueeze(dim=-1)+1)%2,mask.unsqueeze(dim=-1)],dim=2).float()) if mask is not None else None
        sample['bin_label'] = label
        sample['bin_label_onehot'] = self.label_to_one_hot(label)
        return sample

    def label_to_one_hot(self, x, class_count = 2):
        return torch.eye(class_count)[x]