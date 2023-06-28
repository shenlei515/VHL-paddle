import os
import logging

import paddle

import pathlib
import warnings
import math
import numpy as np
from PIL import Image
from typing import Union, Optional, List, Tuple, Text, BinaryIO

def setup_checkpoint_config(args):
    # assert args.checkpoint_save is True
    save_checkpoints_config = {}
    save_checkpoints_config["model_state_dict"] = True if args.checkpoint_save_model else False
    save_checkpoints_config["optimizer_state_dict"] = True if args.checkpoint_save_optim else False
    save_checkpoints_config["train_metric_info"] = True if args.checkpoint_save_train_metric else False
    save_checkpoints_config["test_metric_info"] = True if args.checkpoint_save_test_metric else False
    save_checkpoints_config["checkpoint_root_path"] = args.checkpoint_root_path
    save_checkpoints_config["checkpoint_epoch_list"] = args.checkpoint_epoch_list
    save_checkpoints_config["checkpoint_file_name_save_list"] = args.checkpoint_file_name_save_list
    save_checkpoints_config["checkpoint_file_name_prefix"] = setup_checkpoint_file_name_prefix(args)
    return save_checkpoints_config


def setup_checkpoint_file_name_prefix(args):
    checkpoint_file_name_prefix = ""
    for i, name in enumerate(args.checkpoint_file_name_save_list):
        checkpoint_file_name_prefix += str(getattr(args, name))
        if i != len(args.checkpoint_file_name_save_list) - 1:
            checkpoint_file_name_prefix += "-"
    return checkpoint_file_name_prefix

def setup_save_checkpoint_common_name(save_checkpoints_config, extra_name=None):
    if extra_name is not None:
        checkpoint_common_name = "checkpoint-" + extra_name + "-" \
            + save_checkpoints_config["checkpoint_file_name_prefix"]
    else:
        checkpoint_common_name = "checkpoint-" \
            + save_checkpoints_config["checkpoint_file_name_prefix"]

    return checkpoint_common_name


def setup_save_checkpoint_path(save_checkpoints_config, extra_name=None, epoch="init", postfix=None):
    # if extra_name is not None:
    #     checkpoint_path = save_checkpoints_config["checkpoint_root_path"] \
    #         + "checkpoint-" + extra_name + "-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
    #         + "-epoch-"+str(epoch) + ".pth"
    # else:
    #     checkpoint_path = save_checkpoints_config["checkpoint_root_path"] \
    #         + "checkpoint-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
    #         + "-epoch-"+str(epoch) + ".pth"
    if postfix is not None:
        postfix_str = "-" + postfix
    else:
        postfix_str = ""

    checkpoint_common_name = setup_save_checkpoint_common_name(save_checkpoints_config, extra_name=extra_name)
    checkpoint_path = save_checkpoints_config["checkpoint_root_path"] + checkpoint_common_name \
        + "-epoch-"+str(epoch) + postfix_str + ".pth"
    return checkpoint_path 


def save_checkpoint(args, save_checkpoints_config, extra_name=None, epoch="init",
                    model_state_dict=None, optimizer_state_dict=None,
                    train_metric_info=None, test_metric_info=None, check_epoch_require=True,
                    postfix=None):
    if save_checkpoints_config is None:
        logging.info("WARNING: Not save checkpoints......")
        return
    if (check_epoch_require and epoch in save_checkpoints_config["checkpoint_epoch_list"]) \
        or (check_epoch_require is False):
        checkpoint_path = setup_save_checkpoint_path(save_checkpoints_config, extra_name, epoch, postfix)
        if not os.path.exists(save_checkpoints_config["checkpoint_root_path"]):
            os.makedirs(save_checkpoints_config["checkpoint_root_path"])
        paddle.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict if save_checkpoints_config["optimizer_state_dict"] else None,
            'train_metric_info': train_metric_info if save_checkpoints_config["train_metric_info"] else None,
            'test_metric_info': test_metric_info if save_checkpoints_config["test_metric_info"] else None,
            }, checkpoint_path)
        logging.info("WARNING: Saving checkpoints {} at epoch {}......".format(
            checkpoint_path, epoch))
    else:
        logging.info("WARNING: Not save checkpoints......")


def save_checkpoint_without_check(args, save_checkpoints_config, extra_name=None, epoch="init",
                    model_state_dict=None, optimizer_state_dict=None,
                    train_metric_info=None, test_metric_info=None, check_epoch_require=True,
                    postfix=None):
    checkpoint_path = setup_save_checkpoint_path(save_checkpoints_config, extra_name, epoch, postfix=postfix)
    if not os.path.exists(save_checkpoints_config["checkpoint_root_path"]):
        os.makedirs(save_checkpoints_config["checkpoint_root_path"])
    paddle.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict if save_checkpoints_config["optimizer_state_dict"] else None,
        'train_metric_info': train_metric_info if save_checkpoints_config["train_metric_info"] else None,
        'test_metric_info': test_metric_info if save_checkpoints_config["test_metric_info"] else None,
        }, checkpoint_path)
    logging.info("WARNING: Saving checkpoints {} at epoch {}......".format(
        checkpoint_path, epoch))


def load_checkpoint(args, save_checkpoints_config, extra_name, epoch, postfix=None):
    checkpoint_path = setup_save_checkpoint_path(save_checkpoints_config, extra_name, epoch, postfix=postfix)
    if os.path.exists(checkpoint_path):
        checkpoint = paddle.load(checkpoint_path)
        logging.info(f"checkpoint['model_state_dict'].keys():\
            {list(checkpoint['model_state_dict'].keys())}")
        # model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = None
        logging.info(f"path: {checkpoint_path} Not exists!!!!!!!")
    return checkpoint, checkpoint_path


def load_checkpoint_dict(args, epoch_list_name, extra_name):

    epoch_list = getattr(args, epoch_list_name, [0]) 
    save_checkpoints_config = setup_checkpoint_config(args)
    checkpoint_with_epoch = {}
    checkpoint_with_epoch_paths = {}
    for epoch in epoch_list:
        logging.info("Getting epoch %s model ..." % (epoch))
        checkpoint, checkpoint_path = load_checkpoint(args, save_checkpoints_config, extra_name, epoch)
        checkpoint_with_epoch[epoch] = checkpoint
        checkpoint_with_epoch_paths[epoch] = checkpoint_path
    return save_checkpoints_config, checkpoint_with_epoch, checkpoint_with_epoch_paths


def save_images(args, data, nrow=8, epoch=0, extra_name=None, postfix=None):
    extra_name_str = "images-" + extra_name if extra_name is not None else "images-"
    postfix_str = "-" + postfix if postfix is not None else ""
    image_path = args.checkpoint_root_path + \
        extra_name_str + setup_checkpoint_file_name_prefix(args) + \
        "-epoch-"+str(epoch) + postfix_str + '.jpg'
    save_image(tensor=data, fp=image_path, nrow=nrow)


@paddle.no_grad()
def make_grid(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
              nrow: int=8,
              padding: int=2,
              normalize: bool=False,
              value_range: Optional[Tuple[int, int]]=None,
              scale_each: bool=False,
              pad_value: int=0,
              **kwargs) -> paddle.Tensor:
    if not (isinstance(tensor, paddle.Tensor) or
            (isinstance(tensor, list) and all(
                isinstance(t, paddle.Tensor) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clip(min=low, max=high)
            img = img - low
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] +
                                                        padding)
    num_channels = tensor.shape[1]
    grid = paddle.full((num_channels, height * ymaps + padding,
                        width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, x * width + padding:(
                x + 1) * width] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
               fp: Union[Text, pathlib.Path, BinaryIO],
               format: Optional[str]=None,
               **kwargs) -> None:
    grid = make_grid(tensor, **kwargs)
    ndarr = paddle.clip(grid * 255 + 0.5, 0, 255).transpose(
        [1, 2, 0]).cast("uint8").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)