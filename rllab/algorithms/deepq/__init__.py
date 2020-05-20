
from .trainer import train

def execute(args):
    if args.command == 'train':
        train(**args.arguments)()
    else:
        pass

