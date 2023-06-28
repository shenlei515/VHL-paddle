import paddle
import numpy as np


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = paddle.to_tensor(np.asarray(model_params_list[k])).astype(np.float32)
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params
