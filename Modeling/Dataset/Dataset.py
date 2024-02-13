from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from pathlib import Path
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, images_list: list, class_list: list, transform):
        super(CustomDataset, self).__init__()
        self.images_list = images_list   # Create data from folder
        self.transform = transform
        self.class_list = class_list

    def __getitem__(self, idx):
        x = Image.open(self.images_list[idx])
        y = self.class_list.index(self.images_list[idx].parents[0].stem)
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

    def __len__(self):
        return len(self.images_list)

