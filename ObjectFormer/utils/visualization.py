from torch.utils.tensorboard import SummaryWriter

from ObjectFormer.utils.distributed import is_master_proc

class TensorBoardWriter():
    def __init__(self, **kwargs):
        self.is_master_proc = is_master_proc()
        if self.is_master_proc:
            self.writer = SummaryWriter(**kwargs)

    def add_scalar(self, **kwargs):
        if self.is_master_proc:
            self.writer.add_scalar(**kwargs)

    def flush(self):
        if self.is_master_proc:
            self.writer.flush()

    def close(self):
        if self.is_master_proc:
            self.writer.close()
