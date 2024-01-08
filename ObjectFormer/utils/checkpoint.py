#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified in 2022 by Fudan Vision and Learning Lab

import os

import torch

from ObjectFormer.utils.distributed import is_master_proc
import ObjectFormer.utils.logging as logging

logger = logging.get_logger(__name__)


def make_checkpoint_dir(dir):
    if is_master_proc() and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
        logger.info(f"Create {dir} successfully.")


def get_checkpoint_dir(path_to_job):
    return os.path.join(path_to_job, 'checkpoints')


def get_path_to_checkpoint(path_to_job, epoch):
    name = f'ckpt_epoch_{epoch:05d}.pyth'
    return os.path.join(get_checkpoint_dir(path_to_job), name)

def get_last_checkpoint(path_to_job):
    d = get_checkpoint_dir(path_to_job)
    checkpoints = os.listdir(d) if os.path.exists(d) else []
    checkpoints = [f for f in checkpoints if f.startswith('ckpt_epoch_')]
    if len(checkpoints) == 0:
        return None
    else:
        checkpoint = sorted(checkpoints)[-1]
        return os.path.join(d, checkpoint)


# def has_checkpoint(path_to_job):
#     d = get_checkpoint_dir(path_to_job)
#     files = os.listdir(d) if os.path.exists(d) else []
#     return any('checkpoint' in f for f in files)

def is_checkpoint_epoch(cfg, cur_epoch):
    if cur_epoch + 1 == cfg['TRAIN']['MAX_EPOCH']:
        return True
    return (cur_epoch + 1) % cfg['TRAIN']['CHECKPOINT_PERIOD'] == 0


def save_checkpoint(model, optimizer, scheduler, scaler, cur_epoch, cfg):
    if not is_master_proc():
        return
    if not is_checkpoint_epoch(cfg, cur_epoch):
        return
    
    path_to_job = cfg['OUTPUT_DIR']
    make_checkpoint_dir(get_checkpoint_dir(path_to_job))
    state_dict = (
        model.module.state_dict() if cfg['NUM_GPUS'] > 1 else model.state_dict()
    )

    checkpoint = {
        'epoch': cur_epoch,
        'model_state': state_dict,
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'cfg': cfg,
    }
    if scheduler:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    
    path_to_checkpoint = get_path_to_checkpoint(path_to_job, cur_epoch)
    with open(path_to_checkpoint, 'wb') as f:
        torch.save(checkpoint, f)
    logger.info(f'New checkpoint saved: {path_to_checkpoint}')


def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch_reset=False
):
    #assert os.path.exists(
    #    path_to_checkpoint
    #), 'Checkpoint {} not found'.format(path_to_checkpoint)
    if path_to_checkpoint == "":
        logger.info("Testing without checkpoints, most likely debug..")
        return
    
    logger.info('Loading checkpoint from {}.'.format(path_to_checkpoint))

    ms = model.module if data_parallel else model

    with open(path_to_checkpoint, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')

    pre_train_dict = checkpoint['model_state']
    model_dict = ms.state_dict()
    # Match pre-trained weights that have same shape as current model.
    pre_train_dict_match = {  # TODO del?
        k: v
        for k, v in pre_train_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    # Load pre-trained weights.
    missing_keys, unexpected_keys = ms.load_state_dict(pre_train_dict_match, strict=False)
    logger.info('missing keys: {}'.format(missing_keys))
    logger.info('unexpected keys: {}'.format(unexpected_keys))

    epoch = -1
    if not epoch_reset:
        epoch = checkpoint['epoch']
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        if scaler:
            scaler.load_state_dict(checkpoint['scaler_state'])
    return epoch


def load_test_checkpoint(cfg, model):
    #assert (
    #    cfg['TEST']['CHECKPOINT_PATH'] != ''
    #), 'TEST_CHECKPOINT_PATH is empty!'
    load_checkpoint(
        cfg['TEST']['CHECKPOINT_PATH'],
        model,
        cfg['NUM_GPUS'] > 1,
    )


def load_train_checkpoint(model, optimizer, scheduler, scaler, cfg):
    start_epoch = 0
    if cfg['TRAIN']['CHECKPOINT_PATH'] != '':
        logger.info('Load from given checkpoint file: {}'.format(cfg['TRAIN']['CHECKPOINT_PATH']))
        checkpoint_epoch = load_checkpoint(
            cfg['TRAIN']['CHECKPOINT_PATH'],
            model,
            cfg['NUM_GPUS'] > 1,
            optimizer,
            scheduler,
            scaler,
            cfg['TRAIN']['CHECKPOINT_EPOCH_RESET'],
        )
        start_epoch = checkpoint_epoch + 1
    return start_epoch

def resume_train(model, optimizer, scheduler, scaler, cfg):
    logger.info('Try load from last checkpoint.')
    last_checkpoint = get_last_checkpoint(cfg['OUTPUT_DIR'])
    if last_checkpoint:
        checkpoint_epoch = load_checkpoint(
            last_checkpoint,
            model,
            cfg['NUM_GPUS'] > 1,
            optimizer,
            scheduler,
            scaler,
            False,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        logger.info('No checkpoint found. Start new training.')
        start_epoch = 0
    return start_epoch