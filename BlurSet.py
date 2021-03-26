from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import numpy as np
import cv2
import torch
import albumentations as albu
import glob
from PIL import Image, ImageEnhance


class MTransform:
    def __init__(self):
        pass

    @classmethod
    def check(cls,p):
        return np.random.choice([0, 1], size=1, p=[1 - p, p])[0] > 0

    @classmethod
    def ColorJittering(cls,image, target, p=0.5):
        if MTransform.check(p):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            random_factor = np.random.randint(0, 21) / 10.  # 随机因子
            color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
            random_factor = np.random.randint(10, 21) / 10.  # 随机因子
            brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
            random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
            contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
            return np.array(contrast_image), target
        else:
            return image, target

    @classmethod
    def GaussNoise(cls,image, target, p=0.5):
        if MTransform.check(p):
            aug = albu.GaussNoise(var_limit=(20, 100), always_apply=False, p=0.5)
            result = aug(image=image, target=target)
            return result["image"], result["target"]
        else:
            return image, target

    @classmethod
    def Cutout(cls,image, target, mask=None, max_num_holes=3, size=50, p=0.8, fill_value=0):
        cropped = np.ones_like(target).astype(np.float)

        if np.random.choice([0, 1], size=1, p=[1 - p, p])[0] > 0:
            num_holes = np.random.randint(low=1, high=max_num_holes + 1)
            h, w, c = image.shape
            xs = [np.random.randint(0, h) for i in range(num_holes)]
            ys = [np.random.randint(0, w) for i in range(num_holes)]
            for i in range(num_holes):
                cropped[xs[i]:xs[i] + size, ys[i]:ys[i] + size] = 0
                image[xs[i]:xs[i] + size, ys[i]:ys[i] + size, :] = fill_value

        return image, target, cropped if mask is None else cropped*mask

    @classmethod
    def Compose(cls,image, target, mask=None, p=0.5, max_num_noles=3, size=50, fill_value=0):
        image, target = MTransform.ColorJittering(image, target, p)
        image, target = MTransform.GaussNoise(image, target, p)
        return MTransform.Cutout(image, target, mask, max_num_holes=max_num_noles, size=size, fill_value=fill_value, p=p)


def get_transforms(crop_size: int, size: int, validate=False):
    randomCrop = albu.Compose([albu.RandomCrop(crop_size, crop_size, always_apply=True), albu.Resize(size, size),
                             albu.VerticalFlip(), albu.RandomRotate90(always_apply=True)],
                            additional_targets={'target': 'image'})
    centerCrop = albu.Compose([albu.CenterCrop(crop_size, crop_size, always_apply=True), albu.Resize(size, size),
                               albu.VerticalFlip(), albu.RandomRotate90(always_apply=True)],
                              additional_targets={'target': 'image'})
    pipeline = randomCrop if not validate else randomCrop

    def process(a, b, c):
        r = pipeline(image=a, target=b, mask=c)
        return r['image'], r['target'], r['mask']

    return process


class BlurDataSet(Dataset):
    def __init__(self, data_dir, target_dir, crop_dir, aug, multi_scale=False, validate=False, maug=True):
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.crop_dir = crop_dir
        self.aug = aug
        self.maug = maug

        if not os.path.exists(data_dir):
            raise RuntimeError("dataset error:", self.target_dir + 'not exists')
        if not os.path.exists(target_dir):
            raise RuntimeError("dataset error:", self.data_dir + 'not exists')

        self.data_name_list = []
        self.target_name_list = []
        self.crop_name_list = []
        for _, _, file_names in os.walk(self.data_dir):
            for fileName in file_names:
                self.data_name_list.append(os.path.join(self.data_dir, fileName))
        for _, _, file_names in os.walk(self.target_dir):
            for fileName in file_names:
                self.target_name_list.append(os.path.join(self.target_dir, fileName))
        for _, _, file_names in os.walk(self.crop_dir):
            for fileName in file_names:
                self.crop_name_list.append(os.path.join(self.crop_dir, fileName))
        data_len = len(self.data_name_list)
        target_len = len(self.target_name_list)
        crop_len = len(self.crop_name_list)
        if data_len != target_len:
            raise RuntimeError("different num of data and target in " + self.data_dir + ' and ' + self.target_dir)
        self.data_name_list.sort()
        self.target_name_list.sort()
        self.crop_name_list.sort()
        self.transform = get_transforms(256, 224, validate)
        self.multi_scale_transform = [albu.Resize(224, 224), albu.Resize(112, 112), albu.Resize(56, 56), albu.Resize(28, 28)]
        self.multi_scale = multi_scale

    def __getitem__(self, item):
        image = np.array(Image.open(self.data_name_list[item],'r'))
        target = np.array(Image.open(self.target_name_list[item],'r'))
        crop = (np.array(Image.open(self.crop_name_list[item],'r'))/255)
        # print("nptype",crop.dtype)
        if len(target.shape) > 2:
            target = target[:,:,0]
        if len(crop.shape) > 2:
            crop = crop[:,:,0]


        if self.aug:
            check_time = 5
            for i in range(check_time):
                image_t, target_t, cropped = self.transform(image,target, crop)
                # print("to tensor:", np.min(cropped))
                if self.maug:
                    image_t, target_t, cropped = MTransform.Compose(image_t, target_t, mask=cropped, p=0.5)

                if np.max(target_t) != np.min(target_t) or i == check_time - 1:
                    image = image_t
                    target = target_t
                    break
        # print("to tensor type:", cropped)
        if not self.maug:
            cropped = crop
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()/255
        if not self.multi_scale:
            target = torch.from_numpy(target).long()
            return image, target, cropped
        else:
            targets = []
            croppeds = []

            for tran in self.multi_scale_transform:
                resize = tran(image=target)['image']
                resize = torch.from_numpy(resize).long()
                targets.append(resize)
                resize = tran(image=cropped)['image']
                resize = torch.from_numpy(resize)
                croppeds.append(resize)

            return image, targets, croppeds

    def reRangeTensor(self, tensor, rmin, rmax):
        org_min = torch.min(tensor)
        org_max = torch.max(tensor)
        tensor = (tensor-org_min)/(org_max-org_min) * (rmax-rmin) + rmin
        return tensor

    def __len__(self):
        return len(self.data_name_list)


def test_dataset(data_set):
    size = len(data_set)
    for i in range(size):
        a = data_set.__getitem__(i)
        b = np.transpose(a[0].numpy(), (1, 2, 0))
        c = a[1][0].numpy().astype(np.uint8) * 120
        cv2.imshow('img', b)
        cv2.imshow('mask', c)
        cv2.waitKey(0)


class DataSetManager:
    def __init__(self, opt):
        self.cross_valid = opt.cross_validate
        self.train_data_dir = opt.train_A
        self.train_target_dir = opt.train_B
        self.train_crop_dir = opt.train_C
        self.valid_data_dir = opt.valid_A
        self.valid_target_dir = opt.valid_B
        self.valid_crop_dir = opt.valid_C
        self.aug = opt.aug
        self.batch_size = opt.batch_size
        self.validset_size = len(glob.glob(os.path.join(self.valid_data_dir, "*")))
        self.trainset_size = len(glob.glob(os.path.join(self.train_data_dir, "*")))
        self.whole_size = self.validset_size + self.trainset_size
        self.interval = opt.cross_interval
        self.multi_scale = "MS" in opt.loss_type

    def initialDatasets(self):
        self.train = BlurDataSet(self.train_data_dir, self.train_target_dir, self.train_crop_dir, self.aug, self.multi_scale)
        self.valid = BlurDataSet(self.valid_data_dir, self.valid_target_dir, self.valid_crop_dir, self.aug, self.multi_scale, validate=True, maug=False)
        return self.train, self.valid

    def dataset_cross(self):
        self.train, self.valid = self.cross_sample()
        self.train_loader = DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(dataset=self.valid, batch_size=self.batch_size, shuffle=True)

    def cross_sample(self):
        old_train_data_list = self.train.data_name_list
        old_train_target_list = self.train.target_name_list
        old_train_crop_list = self.train.crop_name_list

        old_valid_data_list = self.valid.data_name_list
        old_valid_target_list = self.valid.target_name_list
        old_valid_crop_list = self.valid.crop_name_list

        new_valid_index = np.random.choice(self.trainset_size, self.validset_size, replace=False)
        old_train_left_index = np.delete(range(self.trainset_size), new_valid_index)

        new_valid_data_list = (np.array(old_train_data_list)[new_valid_index]).tolist()
        new_valid_target_list = (np.array(old_train_target_list)[new_valid_index]).tolist()
        new_valid_crop_list = (np.array(old_train_crop_list)[new_valid_index]).tolist()
        
        new_train_data_list = (np.array(old_train_data_list)[old_train_left_index]).tolist() + old_valid_data_list
        new_train_target_list = (np.array(old_train_target_list)[old_train_left_index]).tolist() + old_valid_target_list
        new_train_crop_list = (np.array(old_train_crop_list)[old_train_left_index]).tolist() + old_valid_crop_list
        

        train = BlurDataSet(self.train_data_dir, self.train_target_dir, self.train_crop_dir, self.aug, self.multi_scale, maug=False)
        valid = BlurDataSet(self.valid_data_dir, self.valid_target_dir, self.valid_crop_dir,self.aug, self.multi_scale, validate=True, maug=False)

        train.data_name_list = new_train_data_list
        train.target_name_list = new_train_target_list
        train.crop_name_list = new_train_crop_list
        valid.data_name_list = new_valid_data_list
        valid.target_name_list = new_valid_target_list
        valid.crop_name_list = new_valid_crop_list


        return train, valid


    def initial(self):
        train, valid = self.initialDatasets()
        self.train_loader = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(dataset=valid, batch_size=self.batch_size, shuffle=True)

    def update_set(self, epoch):
        if epoch % self.interval == 0 and epoch > 0 and self.cross_valid:
            self.dataset_cross()
            print("cross train and valid dataset")

if __name__ == '__main__':
    exit(0)
