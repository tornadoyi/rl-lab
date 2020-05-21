import os
from torch.multiprocessing import Process
import torch.distributed as dist


def _on_process_launch(
        rank, world_size, backend, init_method, timeout, store, group_name,
        target, args, kwargs
):
    # init process group
    grp_args = {
        'rank':             rank,
        'backend':          backend,
        'world_size':       world_size,
        'init_method':      init_method,
        'store':            store,
    }
    if timeout is not None: grp_args['timeout'] = timeout
    if group_name is not None: grp_args['group_name'] = group_name
    dist.init_process_group(**grp_args)

    # call target
    target(*args, **kwargs)


def launch(
        world_size=-1,
        rank_start=0,
        rank_end=None,
        backend='gloo',
        method=None,
        timeout=None,
        store=None,
        group_name=None,
        target=None,
        args=(),
        kwargs={},
):
    # check
    if not dist.is_available(): raise Exception('Distributed is not available')
    if method == None or method == 'env://':
        address, port = os.environ.get('MASTER_ADDR', None), os.environ.get('MASTER_PORT', None)
        if address is None: raise Exception('MASTER_ADDR should be set in environment')
        if port is None: raise Exception('MASTER_PORT should be set in environment')

    if world_size < 0: world_size = os.environ.get('WORLD_SIZE', -1)
    if world_size < 0: raise Exception('Invalid world size {}'.format(world_size))
    rank_end = rank_end or world_size
    if rank_start >= rank_end: raise Exception('invalid rank range {}'.format((rank_start, rank_end)))
    if target is None: raise Exception('invalid target {}'.format(target))
    if backend == 'gloo':
        if not dist.is_gloo_available(): raise Exception('backend gloo is not available')
    elif backend == 'nccl':
        if not dist.is_nccl_available(): raise Exception('backend nccl is not available')
    elif backend == 'mpi':
        if not dist.is_mpi_available(): raise Exception('backend mpi is not available')
    else:
        raise Exception('invalid backend {}'.format(backend))


    # launch process
    processes = []
    for rank in range(rank_start, rank_end, 1):
        p = Process(
            target=_on_process_launch,
            args=(
                rank, world_size, backend, method, timeout, store, group_name,
                target, args, kwargs,
            )
        )
        p.start()
        processes.append(p)

    # join
    for p in processes:
        p.join()