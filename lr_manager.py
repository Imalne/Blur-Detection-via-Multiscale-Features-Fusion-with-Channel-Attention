class StageLRManager:
    def __init__(self, opt, optimizer):
        self.lr_stages = opt.lr_stages
        self.epoch_stages = opt.epoch_stages
        self.optimizer = optimizer

    def lr_step(self, epoch):
        lr = self.getstagelr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr

    def getstagelr(self, epoch):
        assert len(self.lr_stages) == len(self.epoch_stages)
        for i in range(len(self.lr_stages)):
            if epoch == self.epoch_stages[i]:
                return self.lr_stages[i]
        return 1

    def updateValidLog(self, valid_loss):
        pass

class LogLRManager:
    def __init__(self, opt, optimizer):
        self.min_lr = opt.min_lr
        self.max_lr = opt.max_lr
        self.best_valid = 10
        self.valid_up_count = 0
        self.valid_up_times = opt.vutimes
        self.lr_reduce_rate = opt.lr_re_rate
        self.optimizer = optimizer

    def updateValidLog(self, valid_loss):
        if self.best_valid > valid_loss:
            self.best_valid = valid_loss
            self.valid_up_count = 0
        else:
            self.valid_up_count += 1

    def lr_step(self, epoch):
        if self.valid_up_times > self.valid_up_count:
            pass
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.lr_reduce_rate, self.min_lr)
            self.valid_up_count -= 10
