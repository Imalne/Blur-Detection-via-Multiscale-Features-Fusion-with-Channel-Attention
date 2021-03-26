from model.net import *
from model.modules import *
from model.outlayer import *
from model.unit import *
import os
from torch.optim import *
from lr_manager import *


def getModel(opt, log=None, save_path=None, mode="train"):
    if opt.model_type == "re3":
        net = IDEARe_MScale(opt.backbone_type, opt.recurrent_time, opt.class_num,
                            getUnitByType(opt.reunit_type, opt.reunit_skipconnect),
                            getOutLayerByType(opt.outlayer_type), dropout_possibility=opt.dropout_probs)
    elif opt.model_type == "re_sk":
        net = IDEA_MScale_SK(opt.backbone_type, opt.class_num,
                            getOutLayerByType(opt.outlayer_type))
    elif opt.model_type == "re_sk_mo":
        net = IDEA_MScale_SK_MOL(opt.backbone_type, opt.class_num,
                            getOutLayerByType(opt.outlayer_type))
    elif opt.model_type == "re_sk_ms":
        net = IDEA_MScale_SK_MS(opt.backbone_type, opt.class_num,
                            getOutLayerByType(opt.outlayer_type))
    elif opt.model_type == "re_sk_msl":
        net = IDEA_MScale_SK_MSL(opt.backbone_type, opt.class_num,
                            getOutLayerByType(opt.outlayer_type))
    else:
        raise RuntimeError("model_type invalid: "+opt.model_type)

    epoch = 0
    if log is not None and os.path.exists(log.last_save_name):
        print("load from save")
        data = torch.load(log.last_save_name)
        net.load_state_dict(data["model"])
        epoch = data['epoch']
    elif save_path is not None:
        print("load from save")

        data = torch.load(save_path)
        params = data["model"]
        net.load_state_dict(params)
        epoch = data['epoch']

    if opt.gpu:
        net = net.cuda()

    return net, epoch


def getOptim(opt, net):
    if opt.optimizer == "Adam":
        return Adam(params=net.parameters(), lr=opt.max_lr)

def getScheduler(opt, optimizer):
    if opt.lr_manage_type == "stage":
        return StageLRManager(opt, optimizer)
    elif opt.lr_manage_type == "log":
        return LogLRManager(opt, optimizer)
    else:
        raise RuntimeError("no such lr_manage_type")


def getImage(out):
    if type(out) == type([]):
        return out[-1]


def getCudaData(data, use_gpu=True):
    if not use_gpu:
        return data
    cropped = None
    inp = data[0].cuda()
    if type(data[1]) == type([]):
        tar = [i.cuda() for i in data[1]]
        if data[2] is not None:
            cropped = [i.cuda() for i in data[2]]
    else:
        tar = data[1].cuda()
        if data[2] is not None:
            cropped = data[2].cuda()
    return inp, tar, cropped
