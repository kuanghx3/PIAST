import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())

from myfunctions import seed_torch

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument('--data_path', type=str,
                    default="./datasets/247data/",
                    help='The directory containing the EV charging data.')
parser.add_argument('--p_data_path', type=str,
                    default="./datasets/57data/",
                    help='The directory containing the EV charging data with prices that change over time.')
parser.add_argument('--LOOK_BACK', type=int, default=12,
                    help='Number of time step of the Look Back Mechanism.')
parser.add_argument('--predict_time', type=int, default=12,
                    help='Number of time step of the predict time.')
parser.add_argument('--nodes', type=int, default=247,
                    help='Number of areas.')
parser.add_argument('--p_nodes', type=int, default=57,
                    help='Number of areas lots with prices that change over time.')
parser.add_argument('--max_epochs_1', type=int, default=300,
                    help='The max meta-training epochs.')
parser.add_argument('--max_epochs_2', type=int, default=200,
                    help='The max fine-tuning epochs.')
parser.add_argument('--max_epochs_3', type=int, default=500,
                    help='The max fine-tuning epochs.')
parser.add_argument('--patience', type=int, default=100,
                    help='The patience of early stopping.')
parser.add_argument('--training_rate', type=float, default=0.7,
                    help='The rate of training set.')
parser.add_argument('--MLP_hidden', type=int, default=64,
                    help='Hidden size of MLP.')
parser.add_argument('--alpha', type=float, default=0.7,
                    help='Alpha.')
parser.add_argument('--e_begin', type=float, default=0.45,
                    help='Alpha2.')
parser.add_argument('--layer', type=float, default=3,
                    help='number of HALSTM input feature.')
parser.add_argument('--kcnn', type=float, default=2,
                    help='kcnn in conv2d.')


args = parser.parse_args(args=[]) # jupyter
# args = parser.parse_args()      # python

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
seed_torch(2023)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
