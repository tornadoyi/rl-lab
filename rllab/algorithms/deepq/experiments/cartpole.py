import sys
import argparse
from rllab import cli

MODULE = __file__.split('/')[-3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='experiment')
    parser.add_argument("cmd", choices=['train', 'play'])
    cmd = parser.parse_args().cmd
    argv = [MODULE, cmd]
    if cmd == 'train':
        argv += [
            'env.id="CartPole-v0"',
            'total_steps=100000',
            'optimizer={"name":"Adam","lr":1e-3}',
            'rb.size=50000',
            'explore.fraction=0.1', 'explore.final=0.02',
        ]
    else:
        argv += [

        ]

    sys.argv = [sys.argv[0]] + argv
    cli.main()