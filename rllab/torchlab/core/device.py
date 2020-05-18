import torch
from rllab.torchlab import cuda

def select_device(name=None, cuda_first=True, gpu_field='memory.free', field_sorted_reversed=True):
    # check name
    if name is not None and name != 'cpu' and name[:4] != 'cuda': raise Exception('Invalid device name {}'.format(name))

    # cpu
    if name == 'cpu': return torch.device('cpu')

    # cuda
    if name[:4] == 'cuda':
        # check cuda
        if not torch.cuda.is_available(): raise Exception('cuda is not available')

        # specific cuda
        id = name[5:]
        if len(id) > 0: return torch.device(name)

        # select cuda automatically
        id = cuda.sort_gpu(gpu_field, field_sorted_reversed)[0]
        return torch.device('cuda:{}'.format(id))

    # select automatically
    if not cuda_first or not torch.cuda.is_available(): return torch.device('cpu')
    id = cuda.sort_gpu(gpu_field, field_sorted_reversed)[0]
    return torch.device('cuda:{}'.format(id))
