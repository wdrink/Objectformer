import json
import os
import numpy as np
import torch

from ObjectFormer.utils.checkpoint import load_train_checkpoint,resume_train,save_checkpoint
from ObjectFormer.utils.distributed import init_process_group
import ObjectFormer.utils.logging as logging
from ObjectFormer.utils.build_helper import (
    build_dataloader,
    build_dataset,
    build_loss_fun,
    build_model,
)
from ObjectFormer.utils.meters import EpochTimer, MetricLogger, SmoothedValue
from ObjectFormer.utils.optimizer import build_optimizer
from ObjectFormer.utils.scheduler import build_scheduler
from ObjectFormer.utils.visualization import TensorBoardWriter

from tools.test import perform_test

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, detection_loss_func, localization_loss_func, optimizer, scaler, cur_epoch, cfg, writer=None
):
    model.train()
    train_meter = MetricLogger(delimiter='  ')
    train_meter.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(cur_epoch)
    print_freq = 10

    cur_iter = 0
    for samples in train_meter.log_every(train_loader, print_freq, header):
        optimizer.zero_grad()
        samples = dict(map(lambda sample:(sample[0],sample[1].cuda(non_blocking=True)),samples.items()))
        # with torch.cuda.amp.autocast(enabled=cfg['TRAIN']['AMP_ENABLE']):
        with torch.autograd.set_detect_anomaly(True):
            outputs = model(samples)
            
            detection_loss = 0.0
            localization_loss = 0.0 
            if isinstance(outputs[0], torch.Tensor):
                detection_loss += detection_loss_func(samples['bin_label'].to(outputs[0].dtype), outputs[0])
            else:
                for p in outputs[0]:
                    detection_loss += detection_loss_func(samples['bin_label'].to(p.dtype), p)
            
            if isinstance(outputs[1], torch.Tensor):
                localization_loss += localization_loss_func(samples['mask'], outputs[1])
            else:
                for p in outputs[1]:
                    localization_loss += localization_loss_func(samples['mask'], p)
            
            loss = detection_loss + localization_loss
        
        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        scaler.scale(loss).backward(create_graph=second_order)
        scaler.unscale_(optimizer)
        if 'CLIP_GRAD_L2NORM' in cfg:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg['CLIP_GRAD_L2NORM'])
        elif 'CLIP_GRAD_VAL' in cfg:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg['CLIP_GRAD_VAL'])
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        train_meter.update(loss=loss.item(), detection_loss = detection_loss.item(), localization_loss=localization_loss.item())
        train_meter.update(lr=optimizer.param_groups[0]['lr'])

        if writer:
            global_iter = len(train_loader) * cur_epoch + cur_iter
            writer.add_scalar(tag='train loss', scalar_value=loss.item(), global_step=global_iter)
            writer.add_scalar(tag='lr', scalar_value=optimizer.param_groups[0]['lr'], global_step=global_iter)

    train_meter.synchronize_between_processes()
    logger.info('Averaged stats:' + str(train_meter))


def train(local_rank, cfg):
    init_process_group(local_rank, cfg['NUM_GPUS'])
    np.random.seed(cfg['RNG_SEED'])
    torch.manual_seed(cfg['RNG_SEED'])
    logging.setup_logging(cfg)
    logger.info(json.dumps(cfg, indent=4,ensure_ascii=False, sort_keys=False,separators=(',', ':')))

    log_dir = os.path.join(cfg["OUTPUT_DIR"], "logs")
    writer = TensorBoardWriter(log_dir=log_dir)

    model = build_model(cfg)
    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler, total_epochs = build_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['TRAIN']['AMP_ENABLE'])

    if cfg['TRAIN']['AUTO_RESUME']:
        start_epoch = resume_train(model, optimizer, scheduler, scaler, cfg)
    else:
        start_epoch = load_train_checkpoint(model, optimizer, scheduler, scaler, cfg)

    detection_loss_func, localization_loss_func = build_loss_fun(cfg)
    train_dataset = build_dataset('train', cfg)
    train_loader = build_dataloader(train_dataset, 'train', cfg)
    val_dataset = build_dataset('val', cfg)
    val_loader = build_dataloader(val_dataset, 'val', cfg)
    logger.info("Start epoch: {}".format(start_epoch))
    epoch_timer = EpochTimer()

    for cur_epoch in range(start_epoch, cfg['TRAIN']['MAX_EPOCH']):
        train_loader.sampler.set_epoch(cur_epoch)
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            detection_loss_func,
            localization_loss_func,
            optimizer,
            scaler,
            cur_epoch,
            cfg,
            writer,
        )
        epoch_timer.epoch_toc()
        if (cur_epoch+1) % cfg['TRAIN']['EVAL_PERIOD'] == 0:
            perform_test(val_loader, model, cfg, cur_epoch, writer, mode='Val')
        logger.info(
            f'Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs '
            f'from {start_epoch} to {cur_epoch} take '
            f'{epoch_timer.avg_epoch_time():.2f}s in average and '
            f'{epoch_timer.median_epoch_time():.2f}s in median.'
        )
        logger.info(
            f'For epoch {cur_epoch}, each iteraction takes '
            f'{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. '
            f'From epoch {start_epoch} to {cur_epoch}, each iteraction takes '
            f'{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average.'
        )

        scheduler.step(cur_epoch)
        save_checkpoint(model, optimizer, scheduler, scaler, cur_epoch, cfg)

    writer.flush()
    writer.close()