import torch
from rllab.torchlab import cuda


def select_device(device=None, cuda_first=True, gpu_field='memory.free', field_sorted_reversed=True):
    """
    Deprecated
    """
    # check device
    if device is not None and device != 'cpu' and device[:4] != 'cuda': raise Exception('Invalid device {}'.format(device))

    # cpu
    if device == 'cpu': return torch.device('cpu')

    # cuda
    if device is not None and device[:4] == 'cuda':
        # check cuda
        if not cuda.detect_available(): raise Exception('cuda is not available')

        # specific cuda
        id = device[5:]
        if len(id) > 0: return torch.device(device)

        # select cuda automatically
        id = cuda.nvsmi_sort(gpu_field, field_sorted_reversed)[0]
        return torch.device('cuda:{}'.format(id))

    # select automatically
    if not cuda_first or not cuda.detect_available(): return torch.device('cpu')
    id = cuda.nvsmi_sort(gpu_field, field_sorted_reversed)[0]
    return torch.device('cuda:{}'.format(id))
