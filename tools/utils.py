import argparse
import torch
import yaml

import ObjectFormer.utils.logging as logging
from ObjectFormer.utils.checkpoint import get_path_to_checkpoint

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        required=True,
        type=str,
    )
    return parser.parse_args()


def merge_a_into_b(a, b):
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(
                b[k], dict
            ), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


def load_config(args):
    with open('./configs/default.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    with open(args.cfg_file, 'r') as file:
        custom_cfg = yaml.safe_load(file)
    
    merge_a_into_b(custom_cfg, cfg)
    logger.info(cfg)
    return cfg


def launch_func(cfg, func):
    if cfg['NUM_GPUS'] > 1:
        torch.multiprocessing.spawn(
            func,
            nprocs=cfg['NUM_GPUS'],
            args=(cfg,),
        )
    else:
        func(0, cfg)