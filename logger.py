import torch
from tensorboardX import SummaryWriter
import numpy as np
import os

class MLogger:
    def __init__(self, opt):
        self.save_dir = opt.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(self.save_dir)
        if opt.save_name is not None:
            self.last_save_name = os.path.join(self.save_dir, opt.save_name)
        else:
            self.last_save_name = os.path.join(self.save_dir, 'last_net_' + opt.model_type + '_epoch=' + str(opt.epoch_num) + '.pth')

        self.best_save_name = os.path.join(self.save_dir, 'best_net_' + opt.model_type + '_epoch=' + str(opt.epoch_num) + '.pth')
        self.stage_save_name = os.path.join(self.save_dir, 'stage_net_' + opt.model_type + '_epoch=')
        self.progress = ""
        self.best_metrics = 10
        self.best_start_ech = opt.best_start



    def saveLast(self, net, epoch):
        torch.save({
            "model": net.state_dict(),
            "epoch": epoch
        }, self.last_save_name)

    def saveBest(self, net, epoch):
        torch.save({
            "model": net.state_dict(),
            "epoch": epoch
        }, self.best_save_name)

    def saveStage(self,net, epoch):
        torch.save({
            "model": net.state_dict(),
            "epoch": epoch
        }, self.stage_save_name + str(epoch) + ".pth")

    def train(self):
        self.progress = "train"

    def valid(self):
        self.progress = "valid"

    def getTarandPred(self, tar, pred, crop=None):
        if type(tar) == type([]):
            tar = [1 - i[0].cpu().unsqueeze(0) for i in tar]
            pred = [torch.sigmoid(i[0][0].cpu().detach()).unsqueeze(0) for i in pred]
            if crop is not None:
                crop = crop[0:len(pred)]
                crop = [i[0].cpu().unsqueeze(0) for i in crop]
        else:
            tar = 1 - tar[0].cpu().unsqueeze(0)
            pred = torch.sigmoid(pred[0][0].cpu().detach()).unsqueeze(0)
            if crop is not None:
                crop = crop[0].cpu().unsqueeze(0)
        return tar, pred, crop


    def add_metrics(self, inp, tar, pred, loss, split_loss, ech, edge=None, ap=None):
        tar, pred, edge = self.getTarandPred(tar, pred, edge)
        if self.progress == "":
            return
        self.writer.add_scalar(self.progress + "/loss", loss, global_step=ech)
        if len(split_loss) > 1:
            for name, loss in split_loss:
                self.writer.add_scalar(self.progress + "/" + name, loss, global_step=ech)
        self.writer.add_image(self.progress + "/input image", inp, global_step=ech)
        self.add_predict(pred, ech)
        self.add_target(tar, ech, pred)
        if edge is not None:
            self.add_edge_weight(edge,ech)
        if ap is not None:
            self.writer.add_scalar(self.progress + "/AP", ap, global_step=ech)

    def add_target(self, tar, ech, pred):
        if type(tar) == type([]):
            for i in range(len(pred)):
                self.writer.add_image(self.progress+"/target image/" +str(i), tar[i], global_step=ech)
        else:
            self.writer.add_image(self.progress + "/target image", tar, global_step=ech)

    def add_predict(self, pred, ech):
        # print("pred",pred[0].shape)
        if type(pred) == type([]):
            for i in range(len(pred)):
                self.writer.add_image(self.progress+"/predict image/" +str(i), pred[i], global_step=ech)
        else:
            self.writer.add_image(self.progress + "/predict image", pred, global_step=ech)

    def add_edge_weight(self, edge, ech):
        if type(edge) == type([]):
            for i in range(len(edge)):
                self.writer.add_image(self.progress+"/edge_weight/" +str(i), edge[i], global_step=ech)
        else:
            self.writer.add_image(self.progress + "/edge_weight", edge, global_step=ech)

    def add_lr(self, lr, ech):
        self.writer.add_scalar('learning_rate', lr, global_step=ech)

    def update_best_model(self, net, loss, epoch):
        if loss < self.best_metrics and epoch > self.best_start_ech:
            self.best_metrics = loss
            self.saveBest(net, epoch)


