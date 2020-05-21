import os
import sys
import subprocess
from easydict import EasyDict as edict
from rllab.torchlab.utils import shell

def nvsmi_query(*fileds, tree_format=False):
    # get primitive information
    cmd = 'nvidia-smi --format=csv,noheader,nounits --query-gpu={}'.format(','.join(fileds))
    lines = shell.run(cmd).split('\n')[:-1]

    # parse infos
    status = []
    for i in range(len(lines)):
        s = edict()
        texts = lines[i].split(',')
        assert len(texts) == len(fileds), "len(texts):{} != len(fileds):{}".format(len(texts), len(fileds))
        for j in range(len(fileds)):
            f = fileds[j]
            t = texts[j].strip(' ')
            if tree_format:
                d = s
                keys = f.split('.')
                for k in range(len(keys)):
                    key = keys[k]
                    if k == len(keys)-1: d[key] = t
                    else:
                        if key in d: d = d[key]
                        else:
                            d[key] = edict()
                            d = d[key]
            else:
                s[f] = t

        status.append(s)

    # mapping
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        vis_status = [status[int(id)] for id in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        status = vis_status

    return status


def nvsmi_sort(filed, reverse=True):
    values = [int(s[filed]) for s in nvsmi_query(filed)]
    ids = list(range(len(values)))
    return sorted(ids, key=lambda id: int(values[id]), reverse=reverse)


_CUDA_AVAILABLE = None
def detect_available():
    """
    torch.cuda.is_available() is going to initialize all cuda device. To the disadvantage of multiprocessing,
    cause an runtime error "Cannot re-initialize CUDA in forked subprocess" would be raised.
    :return: bool   cuda available
    """
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None: return _CUDA_AVAILABLE
    _CUDA_AVAILABLE = shell.run('{} -c "import torch;print(torch.cuda.is_available())"'.format(sys.executable)).strip('\n') == 'True'
    return _CUDA_AVAILABLE