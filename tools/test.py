import json
import numpy as np
import torch

import torch.nn.functional as F

from ObjectFormer.utils.checkpoint import load_test_checkpoint
from ObjectFormer.utils.distributed import init_process_group
from ObjectFormer.utils import logging
from ObjectFormer.utils.build_helper import (
    build_dataloader,
    build_dataset,
    build_loss_fun,
    build_model,
)
from ObjectFormer.utils.meters import MetricLogger,TopKAccMetric,AucMetric,F1Metric

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(
    test_loader, model, cfg, cur_epoch=None, writer=None, mode='Test'
):
    metric_logger = MetricLogger(delimiter='  ')
    header = mode + ':'

    model.eval()

    il_acc1_metrics = TopKAccMetric(k=1, num_gpus=cfg['NUM_GPUS'])
    il_auc_metrics = AucMetric(cfg['NUM_GPUS'])
    pl_auc_metrics = AucMetric(cfg['NUM_GPUS'])
    il_f1_metrics = F1Metric(cfg['NUM_GPUS'])
    pl_f1_metrics = F1Metric(cfg['NUM_GPUS'])

    for samples in metric_logger.log_every(test_loader, 10, header):
        samples = dict(map(lambda sample:(sample[0],sample[1].cuda(non_blocking=True)),samples.items()))
        outputs = model(samples)
        
        il_acc1_metrics.update(samples['bin_label'], outputs[0])
        il_auc_metrics.update(samples['bin_label'], outputs[0])
        
        # print(samples['mask'].reshape(-1).shape, outputs[1][-1].shape)
        pl_auc_metrics.update(samples['mask'].reshape(-1), outputs[1][-1].reshape(-1))
        il_f1_metrics.update(samples['bin_label'], (outputs[0] > 0.5))
        
        mask_f1_thres = cfg["TEST"]["THRES"]
        mask_bin = (outputs[1][-1] > mask_f1_thres).int()
        pl_f1_metrics.update(samples['mask'].reshape(-1), mask_bin.reshape(-1))
        
        # metric_logger.update(loss=loss.item(), detection_loss = detection_loss.item(), localization_loss=localization_loss.item())

    metric_logger.synchronize_between_processes()
    il_acc1_metrics.synchronize_between_processes()
    il_auc_metrics.synchronize_between_processes()
    pl_auc_metrics.synchronize_between_processes()
    il_f1_metrics.synchronize_between_processes()
    pl_f1_metrics.synchronize_between_processes()

    if writer and cur_epoch is not None:
        writer.add_scalar(tag='Image-level Acc', scalar_value=il_acc1_metrics.acc, global_step=cur_epoch)
        writer.add_scalar(tag='Image-level AUC', scalar_value=il_auc_metrics.auc, global_step=cur_epoch)
        writer.add_scalar(tag='Pixel-level AUC', scalar_value=pl_auc_metrics.auc, global_step=cur_epoch)
        writer.add_scalar(tag='Image-level F1', scalar_value=il_f1_metrics.f1, global_step=cur_epoch)
        writer.add_scalar(tag='Pixel-level F1', scalar_value=pl_f1_metrics.f1, global_step=cur_epoch)

    
    logger.info(
        f'*** Image-level Acc: {il_acc1_metrics.acc:.3f}'
    )
    logger.info(
        f'*** Image-level Auc: {il_auc_metrics.auc:.3f}  Pixel-level Auc: {pl_auc_metrics.auc:.3f}'
    )
    logger.info(
        f'*** Image-level F1: {il_f1_metrics.f1:.3f}  Pixel-level F1: {pl_f1_metrics.f1:.3f}'
    )


def test(local_rank, cfg):
    init_process_group(local_rank, cfg['NUM_GPUS'])
    np.random.seed(cfg['RNG_SEED'])
    torch.manual_seed(cfg['RNG_SEED'])
    logging.setup_logging(cfg, mode='test')
    logger.info(json.dumps(cfg, indent=4,ensure_ascii=False, sort_keys=False,separators=(',', ':')))

    model = build_model(cfg)
    load_test_checkpoint(cfg, model)
    
    test_dataset = build_dataset('test', cfg)
    test_loader = build_dataloader(test_dataset, 'test', cfg)

    logger.info('Testing model for {} iterations'.format(len(test_loader)))

    perform_test(test_loader, model, cfg)