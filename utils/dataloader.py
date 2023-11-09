import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO

class FashionpediaDataset(Dataset):
    def __init__(self, root_path, set_name, transform):
        self.root_path = root_path
        self.set_name = set_name
        self.transform = transform
        
        self.coco = COCO(os.path.join(root_path, 'annotations', f'instances_attributes_{self.set_name}.json'))
        self.image_ids = self.coco.getImgIds()
        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        img = self.load_image(idx)
        annotations = self.load_annotations(idx)
        
        if self.transform:
            img, annotations = self.transform(img, annotations)
        
        return img, annotations
    
    def load_image(self, idx) -> np.array:
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        path = os.path.join(self.root_path, self.set_name, image_info['file_name'])
        image = cv2.imread(path)
        
        return image.astype(np.float32) / 255.
    
    def load_annotations(self, idx) -> np.array:
        annotation_ids = self.coco.getAnnIds(self.image_ids[idx])
        annotations = np.zeros((0, 5))
        
        if len(annotation_ids) == 0:
            return annotations
        
        coco_annotations = self.coco.loadAnns(self.image_ids[idx])
        
        for anno in coco_annotations:
            
            if anno['bbox'][2] < 1 or anno['bbox'][3] < 1:
                continue
            
            annotation = np.zeros((1, 5))
            annotation[0, :4] = anno['bbox']
            annotation[0, 4] = anno['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)
        
        return annotations
    
def collater(batch):
    imgs, annots = zip(*batch)
    