import sys
import os
from rllab import cli
from rllab import define

if __name__ == '__main__':
    cmd, extras = sys.argv[1], sys.argv[2:]
    if cmd == 'train':
        argv = [
            'env.id="BreakoutNoFrameskip-v0"', 'env.frame_stack=True',
            'total_steps=int(1e6)',
            'optimizer={"name":"Adam","lr":1e-3}',
            'rb.size=50000',
            'explore.fraction=0.1', 'explore.final=0.02',
            'device="cpu"',
        ]
    elif cmd == 'play':
        argv = [

        ]
    else: raise Exception('Invalid command {}'.format(cmd))

    sys.argv = [sys.argv[0], define.module(os.path.realpath(__file__)), cmd] + argv + extras
    cli.main()