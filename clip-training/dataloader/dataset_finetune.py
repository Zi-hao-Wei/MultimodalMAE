# dataloader here
from torch.utils.data import Dataset

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from omegaconf import OmegaConf
import os.path as op
import random

from utils import load_from_yaml_file, read_json, load_config_file


import torch
import numpy as np

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])


def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_id_to_img_path[img_id] = file_name
    
    return img_id_to_img_path

def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for caption_info in annotations['annotations']:
        img_id = caption_info['image_id']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []
        
        caption = caption_info['caption']
        img_id_to_captions[img_id].append(caption)
    
    return img_id_to_captions

class CLIP_COCO_dataset(Dataset):
    """CLIP_COCO_dataset. To train CLIP on COCO-Captions."""

    def __init__(self, config, text_tokenizer, context_length=77, input_resolution=224):
        
        super(CLIP_COCO_dataset, self).__init__()

        self.config = config

        annotation_file = self.config.train_annotation_file
        # print("annotation_file : ", annotation_file)
        annotations = read_json(annotation_file)
# 
        self.img_id_to_filename = get_img_id_to_img_path(annotations)
        # print("img_id_to_filename : ", self.img_id_to_filename)

        self.img_id_to_captions = get_img_id_to_captions(annotations)

        self.img_ids = list(self.img_id_to_filename.keys())
        # print("total image ids = ", len(self.img_ids))
        self.total=[]
        shuffled = random.sample(self.img_ids, len(self.img_ids))
        shuffled2 = random.sample(self.img_ids, len(self.img_ids))

        for ids, s1, s2 in zip(self.img_ids,shuffled,shuffled2):
            self.total.append((ids, ids, 1))
            if ids!=s1:
                self.total.append((ids,s1,-1))
            if ids!=s2:
                self.total.append((ids,s2,-1))


        self.img_dir = self.config.train_img_dir
        # print("img dir : ", self.img_dir)

        self.transform = _transform(input_resolution)
        self._tokenizer = text_tokenizer
        self.context_length = context_length
        # print("DataSet INIT")


    def tokenize(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        # print(tokens)
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        mask = torch.zeros((self.context_length + 197, self.context_length + 197))
        mask[(197 + len(tokens)):, :] = -1e9
        mask[:, (197 + len(tokens)):] = -1e9
        return result, mask

    def __len__(self):
    
        return len(self.img_ids)

    def __getitem__(self, idx):
        # print("INDEX",idx)
        
        img_id = self.total[idx][0]
        text_id = self.total[idx][1]
        label = self.total[idx][2]
    
        # randomly pick one caption from the image captions
        text = random.choice(self.img_id_to_captions[img_id])

        img_filename = self.img_id_to_filename[text_id]

        # print(text,img_filename)
        img_path = op.join(self.img_dir, img_filename)
        img = Image.open(img_path)
        img_input = self.transform(img)
        text_input, mask = self.tokenize(text)
        # print(text_input)
        return img_input, text_input, mask, label
