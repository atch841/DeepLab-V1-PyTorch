import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import numpy as np
import cv2
import random
import SimpleITK as sitk
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import zoom

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, split='train_aug', crop_size=321, label_dir_path='SegmentationClassAug', is_scale=True, is_flip=True):
        self.root = '/home/ubuntu/workshops/datasets/voc12/VOCdevkit/VOC2012/'
        self.ann_dir_path = os.path.join(self.root, 'Annotations')
        self.image_dir_path = os.path.join(self.root, 'JPEGImages')
        self.label_dir_path = os.path.join(self.root, label_dir_path) # SegmentationClassAug_Round1
        self.id_path = os.path.join('./list', split + '.txt')

        self.image_ids = [i.strip() for i in open(self.id_path) if not i.strip() == ' ']
        print('%s datasets num = %s' % (split, self.__len__()))

        self.mean_bgr = np.array((104.008, 116.669, 122.675))
        self.split = split
        self.crop_size = crop_size
        self.ignore_label = 255
        self.base_size = None
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        self.is_augment = True
        self.is_scale = is_scale
        self.is_flip = is_flip
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.image_dir_path, image_id + '.jpg')
        label_path = os.path.join(self.label_dir_path, image_id + '.png')
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)

        if self.is_augment:
            image, label = self._augmentation(image, label)
        
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)
        return image_id, image.astype(np.float32), label.astype(np.int64)
    
    def _augmentation(self, image, label):
        # Scaling
        if self.is_scale:
            h, w = label.shape
            if self.base_size:
                if h > w:
                    h, w = (self.base_size, int(self.base_size * w / h))
                else:
                    h, w = (int(self.base_size * h / w), self.base_size)
            scale_factor = random.choice(self.scales)
            h, w = (int(h * scale_factor), int(w * scale_factor))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int64)

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        # print(bbox)

        if self.is_flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        return image, label



class LiTS_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None, tumor_only=False, pseudo=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.pseudo = pseudo
        self.sample_list_ct = os.listdir(base_dir + 'ct/')
        if pseudo:
            self.sample_list_seg = os.listdir(base_dir + 'pseudo/')
        else:
            self.sample_list_seg = os.listdir(base_dir + 'seg/')
        self.sample_list_ct.sort()
        self.sample_list_seg.sort()
        self.data_dir = base_dir
        self.tumor_only = tumor_only

    def __len__(self):
        return len(self.sample_list_ct)

    def __getitem__(self, idx):
        if self.split == "train":
            image_path = self.data_dir + 'ct/' +  self.sample_list_ct[idx]
            if self.pseudo:
                seg_path = self.data_dir + 'pseudo/' +  self.sample_list_seg[idx]
            else:
                seg_path = self.data_dir + 'seg/' +  self.sample_list_seg[idx]
            assert seg_path[seg_path.rfind('/') + 1:].replace('seg', 'ct') == image_path[image_path.rfind('/') + 1:], (image_path, seg_path)
            image = np.load(image_path)
            label = np.load(seg_path)
        else:
            ct = sitk.ReadImage(self.data_dir + 'ct/' + self.sample_list_ct[idx], sitk.sitkInt16)
            seg = sitk.ReadImage(self.data_dir + 'seg/' + self.sample_list_seg[idx], sitk.sitkUInt8)
            image = sitk.GetArrayFromImage(ct)
            label = sitk.GetArrayFromImage(seg)

            image = image.astype(np.float32)
            image = image / 200

            image = ndimage.zoom(image, (1, 0.5, 0.5), order=3)
            label = ndimage.zoom(label, (1, 0.5, 0.5), order=0)

        if not self.pseudo and self.tumor_only:
            label = (label == 2).astype('float32')

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list_ct[idx][:-4]
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_flip(image, label):
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


if __name__ == "__main__":
    dataset = VOCDataset()
    id, image, label = dataset.__getitem__(0)