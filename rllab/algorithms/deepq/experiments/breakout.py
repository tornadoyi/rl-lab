import sys
import os
from rllab import cli
from rllab import define

if __name__ == '__main__':
    cmd, extras = sys.argv[1], sys.argv[2:]
    if cmd == 'train':
        argv = [
            'env.id="BreakoutNoFrameskip-v0"', 'env.frame_stack=True',
            'deepq.gamma=0.99',
            'total_steps=int(1e7)',
            'learning_starts=10000',
            'optimizer={"name":"Adam","lr":1e-4}',
            'rb.size=50000',
            'explore.fraction=0.1', 'explore.final=0.01',
        ]
    elif cmd == 'play':
        argv = [

        ]
    else: raise Exception('Invalid command {}'.format(cmd))

    sys.argv = [sys.argv[0], define.module(os.path.realpath(__file__)), cmd] + argv + extras
    cli.main()