import torch
from torch.utils.data import DataLoader

import ObjectFormer.datasets
import ObjectFormer.models
import ObjectFormer.utils.logging as logging
from ObjectFormer.utils.registries import (
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    MODEL_REGISTRY,
)

logger = logging.get_logger(__name__)


def build_model(cfg):
    model_cfg = cfg['MODEL']
    name = model_cfg['MODEL_NAME']
    logger.info('MODEL_NAME: ' + name)
    model = MODEL_REGISTRY.get(name)(model_cfg)
    assert torch.cuda.is_available(), "Cuda is not available."
    model = model.cuda()
    if cfg['NUM_GPUS'] > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            find_unused_parameters=model_cfg['FIND_UNUSED_PARAMETERS'],
        )

    return model


def build_loss_fun(cfg):
    loss_cfg = cfg['LOSS']
    detection_loss_name = loss_cfg['DETECTION_LOSS']['NAME']
    localization_loss_name = loss_cfg['LOCALIZATION_LOSS']['NAME']
    logger.info('DETECTION_LOSS:\t' + detection_loss_name)
    logger.info('LOCALIZATION_LOSS:\t' + localization_loss_name)
    detection_loss_func = LOSS_REGISTRY.get(detection_loss_name)(loss_cfg['DETECTION_LOSS'])
    localization_loss_func = LOSS_REGISTRY.get(localization_loss_name)(loss_cfg['LOCALIZATION_LOSS'])
    return detection_loss_func, localization_loss_func



def build_dataset(mode, cfg):
    dataset_cfg = cfg['DATASET']
    name = dataset_cfg['DATASET_NAME']
    logger.info('DATASET_NAME: ' + name + '  ' + mode)
    return DATASET_REGISTRY.get(name)(mode, dataset_cfg)


def build_dataloader(dataset, mode, cfg):
    dataloader_cfg = cfg['DATALOADER']
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=True if mode == 'train' else False,
    )
    return DataLoader(
        dataset,
        batch_size=dataloader_cfg['BATCH_SIZE'],
        sampler=sampler,
        num_workers=dataloader_cfg['NUM_WORKERS'],
        pin_memory=dataloader_cfg['PIN_MEM'],
        drop_last=True if mode == 'train' else False,
    )