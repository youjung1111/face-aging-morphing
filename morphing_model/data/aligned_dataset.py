import os
from data.base_dataset import BaseDataset, get_transform
from PIL import Image


class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.dir = opt.dataroot  # 폴더 자체가 dataroot
        self.A_paths = sorted([os.path.join(self.dir, f) for f in os.listdir(self.dir) if '_A.' in f])
        self.B_paths = sorted([os.path.join(self.dir, f) for f in os.listdir(self.dir) if '_B.' in f])
        self.transform = get_transform(opt)

    def __getitem__(self, index):
    A_path = self.A_paths[index]
    B_path = self.B_paths[index]
    A_img = Image.open(A_path).convert('RGB')
    B_img = Image.open(B_path).convert('RGB')

    A = self.transform(A_img)
    B = self.transform(B_img)

    # MorphGAN 모델은 'reals'라는 키로 받아서 씀!
    return {
        'reals': torch.stack([A, B], dim=0),  # [2, C, H, W] 형태로 합침
        'A_paths': A_path,
        'B_paths': B_path
    }

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))
