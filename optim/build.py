import logging

import paddle

from utils.data_utils import scan_model_with_depth
from utils.model_utils import build_param_groups

# Discard
# from .fedprox import FedProx


"""
    args.opt in 
    ["sgd", "adam"]
    --lr
    --momentum
    --clip-grad # wait to be developed
    --weight-decay, --wd
"""



def create_optimizer(args, model=None, params=None, **kwargs):
    if "role" in kwargs:
        role = kwargs["role"]
    else:
        role = args.role

    # params has higher priority than model
    if params is not None:
        params_to_optimizer = params
    else:
        if model is not None:
            params_to_optimizer = model.parameters()
        else:
            raise NotImplementedError
        pass

    if (role == 'server') and (args.algorithm in [
        'FedAvg']):
        if args.server_optimizer == "sgd":
            # optimizer = paddle.optimizer.Momentum(parameters=params_to_optimizer,
            #     learning_rate=args.lr, weight_decay=args.wd, momentum=args.momentum)
            optimizer = paddle.optimizer.Momentum(parameters=filter(lambda p: not p.stop_gradient, params_to_optimizer),
                learning_rate=args.lr, weight_decay=args.wd, momentum=args.momentum)
        elif args.server_optimizer == "adam":
            optimizer = paddle.optimizer.Adam(parameters=filter(lambda p: not p.stop_gradient, params_to_optimizer),
                learning_rate=args.lr, weight_decay=args.wd, amsgrad=True)
        elif args.server_optimizer == "no":
            print(filter(lambda p: not p.stop_gradient, params_to_optimizer))
            optimizer = paddle.optimizer.Momentum(parameters=filter(lambda p: not p.stop_gradient, params_to_optimizer),
                learning_rate=args.lr, weight_decay=args.wd, momentum=args.momentum)
        else:
            raise NotImplementedError
    else:
        if args.client_optimizer == "sgd":
            optimizer = paddle.optimizer.Momentum(parameters=params_to_optimizer,
                learning_rate=args.lr, weight_decay=args.wd, momentum=args.momentum)
        elif args.client_optimizer == "adam":
            raise NotImplementedError
        elif args.client_optimizer == "no":
            optimizer = paddle.optimizer.Momentum(parameters=params_to_optimizer,
                learning_rate=args.lr, weight_decay=args.wd, momentum=args.momentum)
        else:
            raise NotImplementedError

    return optimizer







