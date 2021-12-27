import argparse
import collections
import torch

def auto_namedtuple(arg='auto_namedtuple', **kwargs):
    return collections.namedtuple(arg, tuple(kwargs))(**kwargs)

def parse_args():
    # Command line arguments parser. Described as in their 'help' sections.
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_of_layers", type=int, default=2, help="The number of LSTM layers the model has.")
    parser.add_argument("--hidden_layer_units", type=int, default=200, help="The number of hidden units per layer.")
    parser.add_argument("--dropout", type=float, default=0.5, help="The dropout parameter.")
    parser.add_argument("--weights_uniforming", type=float, default=0.05, help="The weight initialization parameter.")
    parser.add_argument("--batch_size", type=int, default=20, help="The batch size.")
    parser.add_argument("--sequence_length", type=int, default=35, help="The sequence length for bptt.")
    parser.add_argument("--learning_rate", type=float, default=1, help="The learning rate.")
    parser.add_argument("--total_epochs_num", type=int, default=39, help="Total number of epochs for training.")
    parser.add_argument("--first_epoch_modify_lr", type=int, default=6, help="The epoch to start factoring the learning rate.")
    parser.add_argument("--lr_decrease_factor", type=float, default=1.2, help="The factor to decrease the learning rate.")
    parser.add_argument("--max_gradients_norm", type=float, default=5, help="The maximum norm of gradients we impose on training.")
    args = parser.parse_args()

    set_device(args)
    nt = auto_namedtuple(**vars(args))
    nt2 = collections.namedtuple('temp', "num_of_layers, hidden_layer_units, dropout weights_uniforming, batch_size,"
                                   "sequence_length, learning_rate, total_epochs_num, first_epoch_modify_lr,"
                                   " lr_decrease_factor, max_gradients_norm")
    ntt = nt2(2,200,0.5,0.05,20,35,1,39,6,1.2,5)

    print('Parameters of the model:')
    print('Args:', args)
    print("\n")

    return nt


def set_device(args):
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        print("Model will be training on the CPU.\n")
        args.device = torch.device('cpu')
