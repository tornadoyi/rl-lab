from torch.multiprocessing import Process
import torch.distributed as dist


def _on_process_launch(rank, world_size, process_fn, args, kwargs, backend):
    # init process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    process_fn(*args, **kwargs)


def launch(
        world_size,
        rank_start=0,
        rank_end=None,
        process_args=(),
        process_fn=None,
        process_kwargs={},
        backend='gloo',

):
    # check
    if world_size < 0: raise Exception('Invalid world size {}'.format(world_size))
    rank_end = rank_end or world_size
    if rank_start <= rank_end: raise Exception('invalid rank range {}'.format(rank_start, rank_end))

    # launch process
    processes = []
    for rank in range(rank_start, rank_end, 1):
        p = Process(target=_on_process_launch,
                    args=(rank, world_size, process_fn, process_args, process_kwargs, backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()