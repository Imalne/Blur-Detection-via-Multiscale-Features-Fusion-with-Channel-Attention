import numpy as np
import cv2
from glob import glob
import torchvision.transforms as transforms
from PIL import Image
import tqdm
import albumentations as albu
from option import getPredictParser
from utils import *


class Predictor:
    def __init__(self, config):
        assert config.weight_path != ""
        net, _ = getModel(config, save_path=config.weight_path, mode="eval")
        net.eval()
        self.model = net
        self.trans = transforms.Compose([
            transforms.ToTensor()])
        self.out_index = config.recur_time
        self.getImageMode = config.mode
        self.padOrCrop = config.padOrCrop
        self.use_gpu = config.gpu

    def paddingIfNeed(self, img):
        img = np.array(img)
        height, width, _ = img.shape
        padded_height, padded_width, _ = img.shape
        if padded_height % 32 != 0:
            padded_height = (padded_height // 32 + 1) * 32
        if padded_width % 32 != 0:
            padded_width = (padded_width // 32 + 1) * 32
        pad = albu.PadIfNeeded(padded_height, padded_width)
        crop = albu.CenterCrop(height, width)
        img = pad(image=img)["image"]
        return img, crop

    def cropIfNeed(self, img):
        img = np.array(img)
        height, width, _ = img.shape
        cropped_height, cropped_width, _ = img.shape
        if cropped_height % 32 != 0:
            cropped_height = (cropped_height // 32) * 32
        if cropped_width % 32 != 0:
            cropped_width = (cropped_width // 32) * 32
        crop = albu.Crop(x_min=0, y_min=0, x_max=cropped_width, y_max=cropped_height)
        img = crop(image=img)["image"]
        return img, crop

    def transform(self, img):
        if self.padOrCrop == "pad":
            return self.paddingIfNeed(img)
        elif self.padOrCrop == "crop":
            return self.cropIfNeed(img)
        else:
            raise RuntimeError(" padOrCrop invalid")

    def tar_transform(self, tar, crop):
        if self.padOrCrop == "crop":
            return crop(image=tar)["image"]
        elif self.padOrCrop == "pad":
            return tar
        else:
            raise RuntimeError("padOrCrop invalid")

    def getImage(self, output, mode="single"):
        if type(output) == type([]):
            if self.getImageMode == "single":
                assert self.out_index >= 0
                result = output[self.out_index]
                if type(result) == type([]):
                    result = result[0]
                test = torch.softmax(result.cpu().detach()[0], dim=0)[0].unsqueeze(0)
                #test = torch.sigmoid(result[0].cpu().detach())
                test = np.transpose(test, (1, 2, 0))
                return test[:, :, 0]
            elif self.getImageMode == "average":
                result = output[-1]
                if type(result) == type([]):
                    result = result[0]
                # test = torch.sigmoid(result.cpu().detach())
                test = torch.softmax(result.cpu().detach()[0], dim=0)[0].unsqueeze(0)
                test = np.transpose(test, (1, 2, 0))
                return test[:, :, 0]
            else:
                raise RuntimeError("no such mode of output transfer")
        else:
            return output.cpu().detach().numpy()[0]

    def predict(self, inp: str, target: str = None, merge_img=False):
        assert os.path.exists(inp)

        img = Image.open(inp, 'r')
        img_resize, crop = self.transform(img)
        output, output_view = self.predict_(img_resize)
        img_resize = crop(image=img_resize)["image"]
        output = crop(image=output)["image"]*255
        output_view = crop(image=output_view)["image"]
        if target:
            tar = cv2.imread(target)
            tar = self.tar_transform(tar, crop)
            tar_resize = 1 - tar
            tar_resize_view = tar_resize * 255
            return np.hstack(
                (img_resize, tar_resize_view, output_view)) if merge_img else output_view, output, tar_resize
        else:
            return np.hstack(img_resize, output_view) if merge_img else output_view, output, None

    def predict_(self, inp: np.array):
        org = np.array(inp)
        inp = self.trans(inp)
        inp = torch.unsqueeze(inp, 0)
        if self.use_gpu:
            inp = inp.cuda()
        with torch.no_grad():
            output = self.model(inp)
            output = self.getImage(output, self.getImageMode)
        output = output[:, :, np.newaxis]
        output = np.concatenate((output, output, output), axis=-1)
        return output[:, :, 0], output * 255

    def predict_dir(self, img_path, tar_path=None, out_dir="./submit2", merge_img=False):
        img_paths = glob(os.path.join(img_path, "*"))

        os.makedirs(os.path.join(out_dir, 'view'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'gt'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict'), exist_ok=True)
        if tar_path:
            tar_paths = glob(os.path.join(tar_path, "*"))
            assert len(img_paths) == len(tar_paths)

            img_paths.sort()
            tar_paths.sort()
            bar = tqdm.tqdm(zip(img_paths, tar_paths), total=len(img_paths))

            try:
                for img, tar in bar:
                    # print(img,tar)
                    view, output, target = self.predict(img, tar, merge_img=merge_img)

                    cv2.imwrite(os.path.join(out_dir, 'view', os.path.basename(tar)), view)
                    cv2.imwrite(os.path.join(out_dir, 'predict', os.path.basename(tar)), output)
                    cv2.imwrite(os.path.join(out_dir, 'gt', os.path.basename(tar)), target)
            except KeyboardInterrupt:
                bar.close()
                raise
            bar.close()
        else:
            bar = tqdm.tqdm(img_paths, total=len(img_paths))
            try:
                for img in bar:
                    view, output, _ = self.predict(img, merge_img=merge_img)
                    cv2.imwrite(os.path.join(out_dir, 'view', os.path.basename(img)), view)
                    cv2.imwrite(os.path.join(out_dir, 'predict', os.path.basename(img)), output)
            except KeyboardInterrupt:
                bar.close()
                raise
            bar.close()


if __name__ == '__main__':
    parser = getPredictParser()
    args = parser.parse_args()
    predictor = Predictor(args)
    predictor.predict_dir(args.image_dir,
                          args.target_dir,
                          out_dir=args.out_dir,
                          merge_img=args.merge_img)
    exit(0)
