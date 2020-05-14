
from .trainer import Trainer

def execute(args):
    if args.command == 'train':
        Trainer(**args.arguments)()
    else:
        pass

