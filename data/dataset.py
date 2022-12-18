import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import collections
import torchvision

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, file_filter=lambda x: False):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname) or file_filter(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        # cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        # mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        # ret['cond_image'] = cond_image
        # ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, transforms=None, file_filter=lambda x: False):
        imgs = make_dataset(data_root, file_filter=file_filter)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        if transforms is None:
            self.tfs = transforms.Compose([
                    transforms.Resize((image_size[0], image_size[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            self.tfs = transforms
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask(index)
        # cond_image = img * (1. - mask) + mask * torch.randn_like(img).to(mask.device)
        # mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        # ret['cond_image'] = cond_image
        # ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['channel'] = img.shape[-3]
        return ret
    
    def get_collate_fn(self):
        return None

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, index):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class MaskShiftingUncroppingDataset(UncroppingDataset):
    class GsmLoader:
        def __init__(self, device='cpu'):
            self.device = device
        def __call__(self, path):
            return torch.load(path).to(self.device)
    class MyRandomCrop:
        def __init__(self, size) -> None:
            try:
                size[0]
            except:
                size = [size, size]
            self.size = size
            
        def __call__(self, img: torch.Tensor):
            import torch.nn.functional as F
            import random

            padding = [0] * 4
            for i in [-1, -2]:
                if img.shape[i] < self.size[i]:
                    d = self.size[i] - img.shape[i]
                    low = int(d / 2)
                    high = d - low
                    padding[(-i - 1) * 2] = low
                    padding[(-i - 1) * 2 + 1] = high
            img = F.pad(img, padding)
            for i in [-1, -2]:
                assert img.shape[i] >= self.size[i]

            h, w = self.size
            i = random.randint(0, img.shape[-2] - h)
            j = random.randint(0, img.shape[-1] - w)

            if len(img.shape) == 2:
                return img[i: i + h, j: j + h]
            else:
                assert len(img.shape) == 3
                return img[:, i: i + h, j: j + h]



    def _identitiy(x): return x
    def _file_filer(fname): return fname[-4:] == '.gsm'
    def __init__(self, data_root, class_number, mask_config={}, ground_truth_len=-1, mask_queue_len=-1, image_size=[256, 256], mask_compression=None, default_mask_mode='fourdirection', device='cpu'):
        mask_config['mask_mode'] = 'shifting'
        if mask_compression is None:
            mask_compression = MaskShiftingUncroppingDataset._identitiy
        super().__init__(
            os.path.join(data_root, 'groundtruths'), 
            mask_config, 
            ground_truth_len, 
            image_size, 
            loader=MaskShiftingUncroppingDataset.GsmLoader(device), 
            file_filter=MaskShiftingUncroppingDataset._file_filer,
            transforms=MaskShiftingUncroppingDataset.MyRandomCrop(image_size)
        )
        self.device = device
        self.ground_truth_root = os.path.join(data_root, 'groundtruths')
        self.mask_root = os.path.join(data_root, 'masks')

        # 制备真相
        self.gt_identifiers = []
        self.gt_id_to_index = {}
        self.max_channel = 0
        self.mask_queue_len = mask_queue_len
        self._make_mask_sets()

        self.mask_compression = mask_compression
        self.default_mask_mode = default_mask_mode
    def _get_mask_dir(self, gt_id: str):
        return os.path.join(self.mask_root, gt_id)
    def _make_mask_sets(self):
        self.mask_queues = []
        self.mask_counts = []
        for gt_id in ['.'.join(img.rsplit("/")[-1].rsplit("\\")[-1].rsplit('.')[:-1]) for img in self.imgs]:
            self.add_ground_truth(gt_id, None)

    def _get_shifting_mask(self, index):
        gt_id = self.gt_identifiers[index]
        mask_pos = torch.randint(0, len(self.mask_queues[index]), (1, )).item()
        mask_id = self.mask_queues[index][mask_pos]
        mask = torchvision.io.read_image(os.path.join(self._get_mask_dir(gt_id), str(mask_id) + '.jpg')).to(self.device)
        assert len(mask.shape) == 3 and mask.shape[0] == 3
        mask = mask[:1]
        assert len(mask.shape) == 3 and mask.shape[0] == 1
        return mask / 255

    def is_ground_truth(self, ground_truth_id: str):
        return ground_truth_id in self.gt_id_to_index
    def add_ground_truth(self, ground_truth_id: str, ground_truth: torch.Tensor):
        assert not self.is_ground_truth(ground_truth_id), f'{ground_truth_id} already exists'

        if ground_truth is not None:
            gt_path = os.path.join(self.ground_truth_root, ground_truth_id + '.pt')
            torch.save(ground_truth, gt_path)
            self.imgs.append(gt_path)
        
        self.gt_identifiers.append(ground_truth_id)
        self.gt_id_to_index[ground_truth_id] = len(self.gt_identifiers) - 1

        mask_dir = self._get_mask_dir(ground_truth_id)
        if not os.path.isdir(mask_dir):
            os.mkdir(mask_dir)
        assert os.path.isdir(mask_dir), f"unable to create mask directory at {mask_dir} for ground truth {ground_truth_id}"
        self.mask_queues.append(
            collections.deque(
                ['.'.join(img.rsplit("/")[-1].rsplit("\\")[-1].rsplit('.')[:-1]) for img in make_dataset(mask_dir)],
                 self.mask_queue_len if self.mask_queue_len > 0 else None
            )
        )
        if len(self.mask_queues[-1]) > 0:
            self.mask_counts.append(int(self.mask_queues[-1][-1]))
        else:
            self.mask_counts.append(0)


    def add_mask(self, ground_truth_id: str, mask: torch.Tensor):
        """为一个真相添加新的 mask

        :param str ground_truth_id: 真相的名称，可以在 get_ground_truth_identifiers() 中找到
        :param torch.Tensor mask: BatchSize x H x W 或 H x W
        :raises KeyError: 不合法的真相名称
        """
        if ground_truth_id not in self.gt_id_to_index:
            raise KeyError(f"scene {ground_truth_id} not found")
        index = self.gt_id_to_index[ground_truth_id]
        mask_dir = self._get_mask_dir(ground_truth_id)
        assert os.path.isdir(mask_dir), f"expecting {mask_dir} to be a directory holding masks for ground truth {ground_truth_id}"

        if len(mask.shape) == 2:
            masks = mask.unsqueeze(dim=0)
        else:
            assert len(mask.shape) == 3
            masks = mask
        assert len(masks.shape) == 3
        
        for mask in masks:
            mask = self.mask_compression(mask)
            torchvision.utils.save_image(mask, os.path.join(mask_dir, str(self.mask_counts[index]) + '.jpg'))
            self.mask_queues[index].append(self.mask_counts[index])
            self.mask_counts[index] += 1


    def get_ground_truth_identifiers(self):
        """获取不同真相的标识符，这种标识符将会用于为真相增加新的 mask

        :return list[str]: 标识符列表
        """
        return self.gt_identifiers
    def _get_masks(self, ground_truth_id):
        assert self.is_ground_truth(ground_truth_id)
        return list(self.mask_queues[self.gt_id_to_index[ground_truth_id]])
    def get_mask(self, index):
        try:
            return super().get_mask(index)
        except NotImplementedError:
            if self.mask_mode == 'shifting':
                if len(self.mask_queues[index]) <= 0:
                    self.mask_mode = self.default_mask_mode
                    res = super().get_mask(index)
                    self.mask_mode = 'shifting'
                    return res.to(self.device)
                return self._get_shifting_mask(index)
            else:
                raise NotImplementedError(
                    f'Mask mode {self.mask_mode} has not been implemented.')
    def _pad(data):
        max_channel = 0
        for s in data:
            max_channel = max(max_channel, s['channel'])
        sample:dict = data[0]
        batch:dict = {}
        for key in sample.keys():
            res = []
            for s in data:
                if isinstance(s[key], torch.Tensor) and key != 'mask':
                    original = s[key]
                    padded = torch.nn.functional.pad(
                        original,
                        pad = [0, 0, 0, 0, 0, max_channel - original.shape[-3]],
                        mode='constant',
                        value=0
                    )
                    if len(original.shape) == 3:
                        assert (original != padded[:s['channel']]).sum() == 0, (s['channel'], original.shape, max_channel, padded.shape, (original - padded).abs().sum())
                    elif len(original.shape) == 4:
                        assert (original != padded[:, :s['channel']]).sum() == 0
                    else:
                        raise NotImplementedError

                    res.append(padded)
                else:
                    res.append(s[key])    
            if isinstance(sample[key], torch.Tensor):
                res = torch.stack(res, dim=0)
            batch[key] = res
        return batch
    
    def get_collate_fn(self):
        return MaskShiftingUncroppingDataset._pad


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)

