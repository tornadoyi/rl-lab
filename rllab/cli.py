import signal
import argparse
from easydict import EasyDict as edict
from rllab import algorithms as modules
from rllab import define

MODULES = dict([(n, getattr(modules, n)) for n in dir(modules) if hasattr(getattr(modules, n), 'execute')])


def initialize_path(args, dargs):
    # set path
    if args.root_path is not None: define.set_root_path(args.root_path)
    if args.model_path is not None: define.set_model_path(args.model_path)
    if args.profiling_path is not None: define.set_profiling_path(args.profiling_path)
    define.set_experiment_path(dargs.arguments.env.id if args.exp_path is None else args.exp_path)


def parse_args():
    parser = argparse.ArgumentParser(prog='rl-lab', description="Laboratory of reinforcement learning includes games and algorithms.")
    parser.add_argument("module", choices=list(MODULES.keys()), help='supported modules')
    parser.add_argument("command", help='command of module')
    parser.add_argument('arguments', nargs='*', default=[], help='arguments of command')
    parser.add_argument('--root-path', type=str, default=None, help='root path')
    parser.add_argument('--model-path', type=str, default=None, help='model path')
    parser.add_argument('--profiling-path', type=str, default=None, help='profiling path')
    parser.add_argument('--exp-path', type=str, default=None, help='experiment path')
    args = parser.parse_args()

    # parse parameters
    dargs = edict()
    dargs['module'] = args.module
    dargs['command'] = args.command
    dargs['arguments'] = {}

    # get env keys
    mods, vals = [], []
    for a in args.arguments:
        m, v = a.split('=')
        mods.append(m.split('.'))
        vals.append(v)

    # exec code
    code_dict = {}
    exec("parameters = [{}]".format(','.join(vals)), code_dict)
    values = code_dict['parameters']

    for i in range(len(mods)):
        smods = mods[i]
        d = dargs['arguments']
        for j in range(len(smods)):
            n = smods[j]
            if j >= len(smods) - 1: d[n] = values[i]
            else:
                if n not in d: d[n] = {}
                d = d[n]

    # initialize path
    initialize_path(args, dargs)

    return dargs



def main():

    # catch exit signals
    def handle_signals(signum, frame):
        exit(0)
    signal.signal(signal.SIGINT, handle_signals)
    signal.signal(signal.SIGTERM, handle_signals)

    # parse args
    args = parse_args()
    mod = MODULES.get(args.module, None)

    # execute mod
    mod.execute(args)



if __name__ == '__main__':
    main()
