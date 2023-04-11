import torch 
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils import img_transform

class AllDataset(Dataset):
    
    def __init__(self, imgs_list, class_to_int, transforms = None):
        
        super().__init__()
        self.imgs_list    = imgs_list
        self.class_to_int = class_to_int
        self.transforms   = transforms
        
        
    def __getitem__(self, index):
    
        image_path = self.imgs_list[index]
        
        #Reading image
        image = Image.open(image_path).copy()
        
        
        #Retriving class label
        label_num = self.class_to_int[image_path.split("/")[-2]]
        label = np.zeros(len(self.class_to_int))
        label[label_num] = 1
        

        img_tensor   = img_transform(image)
        label_tensor = torch.from_numpy(label)

        return img_tensor , label_tensor
        
        
        
    def __len__(self):
        return len(self.imgs_list)