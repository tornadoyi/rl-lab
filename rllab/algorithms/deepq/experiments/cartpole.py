import sys
import os
from rllab import cli
from rllab import define

if __name__ == '__main__':
    cmd, extras = sys.argv[1], sys.argv[2:]
    if cmd == 'train':
        argv = [
            'env.id="CartPole-v0"',
            'total_steps=100000',
            'optimizer={"name":"Adam","lr":1e-3}',
            'rb.size=50000',
            'explore.fraction=0.1', 'explore.final=0.02',
        ]
    elif cmd == 'play':
        argv = [

        ]
    else: raise Exception('Invalid command {}'.format(cmd))

    sys.argv = [sys.argv[0], define.module(os.path.realpath(__file__)), cmd] + argv + extras
    cli.main()