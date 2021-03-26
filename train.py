import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import warnings
warnings.filterwarnings('ignore')
from logger import MLogger
from option import getTrainParser
from BlurSet import *
from loss.loss import MLoss
from sklearn import metrics
from utils import *


class Trainer():
    def __init__(self, opt):
        self.datasetManager = DataSetManager(opt)
        self.log = MLogger(opt)
        self.net, self.epoch = getModel(opt, self.log)
        self.loss_func = MLoss(opt)
        self.optimizer = getOptim(opt, self.net)
        self.scheduler = getScheduler(opt, self.optimizer)
        self.total_epoch = opt.epoch_num
        self.use_gpu = opt.gpu


    def train(self):
        self.datasetManager.initial()

        for i in range(self.epoch, self.total_epoch):
            self.datasetManager.update_set(i)
            self.run_epoch(loader=self.datasetManager.train_loader, ech=i)
            self.log.add_lr(self.optimizer.param_groups[0]['lr'], i)
            self.log.saveLast(self.net, i)
            valid_loss = self.valid(loader=self.datasetManager.valid_loader, ech=i)
            self.scheduler.updateValidLog(valid_loss)
            self.scheduler.lr_step(i)
            if i % 100 == 0:
                self.log.saveStage(self.net, i)

    def run_epoch(self, loader, ech):
        self.log.train()
        self.net.train()
        for i, data in enumerate(loader):
            self.optimizer.zero_grad()
            inp, tar, cropped = getCudaData(data, self.use_gpu)
            out = self.net(inp)
            out_num = len(out)
            loss, split = self.loss_func(out, tar, cropped, self.net)
            loss.backward()
            self.optimizer.step()
            if i % 20 == 0:
                out = getImage(out)
                self.log.add_metrics(inp=inp[0].cpu(),
                                tar=tar,
                                pred=out,
                                loss=loss / out_num,
                                split_loss=split,
                                ech=ech * len(loader.dataset) + i * loader.batch_size,
                                edge=cropped)
                print(
                    "epoch {:2d}: {:.0f}% \t  lr {:.6f} \t total loss:{:.6f} \t out_num: {:d} \t average loss:{:.6f}".format(
                        ech,
                        i * loader.batch_size / len(loader.dataset) * 100,
                        self.optimizer.param_groups[0]['lr'],
                        (loss.detach().cpu().numpy()),
                        out_num,
                        (loss.detach().cpu().numpy()) / out_num
                    ))

    def valid(self, loader, ech):
        print("valid ...")
        self.log.valid()
        self.net.eval()
        logged = False
        losses = []
        split_losses = []
        aps = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                inp, tar, cropped = getCudaData(data, self.use_gpu)
                out = self.net(inp)
                loss, splits = self.loss_func(out, tar, cropped, self.net)
                losses.append(loss.detach().cpu().numpy() / len(out))
                split_losses.append(splits)
                ap = self.accuracy(out, tar)
                aps.append(ap)
                if not logged:
                    log_inp = inp[0].cpu()
                    log_tar = tar
                    log_pred = getImage(out)
                    logged = True
            splits = []
            for i in range(len(split_losses[0])):
                splits.append((split_losses[0][i][0], np.mean(
                    [split_losses[j][i][1].cpu().detach().numpy() for j in range(len(split_losses))])))
            self.log.add_metrics(inp=log_inp,
                            tar=log_tar,
                            pred=log_pred,
                            loss=np.mean(losses),
                            split_loss=splits,
                            ech=ech,
                            ap=np.mean(ap))

            print("valid loss: {:6f}".format(np.mean(losses)))
            self.log.update_best_model(self.net, np.mean(losses), ech)
            print("update best model")
            return np.mean(losses)

    def accuracy(self, out, tar):
        out = getImage(out)
        APS = []
        # print(out)
        if type(out) == type([]):
            for i in range(out[0].shape[0]):
                img = torch.sigmoid(out[0][i, 0, :, :]).unsqueeze(0).cpu().numpy()
                gt = 1 - tar[0][i].cpu().numpy()
                # print(np.max(img))
                APS.append((metrics.mean_absolute_error(np.ravel(gt), np.ravel(img))))
        else:
            for i in range(out.shape[0]):
                img = torch.sigmoid(out[i, 0, :, :]).unsqueeze(0).cpu().numpy()
                gt = 1 - tar[i].cpu().numpy()
                APS.append((metrics.mean_absolute_error(np.ravel(gt), np.ravel(img))))
        return np.mean(APS)


if __name__ == '__main__':
    parser = getTrainParser()
    opt = parser.parse_args()
    trainer = Trainer(opt)
    trainer.train()
    exit(0)
