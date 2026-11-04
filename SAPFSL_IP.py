import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import shutil
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
from torch.nn.parameter import Parameter
from operator import itemgetter
use_gpu = torch.cuda.is_available()

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 128)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 200)
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 16)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 10000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-m","--test_class_num",type=int, default=16)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')
parser.add_argument("-t","--test_queries_num_per_class",type=int,default=5)
parser.add_argument("-r","--n_dataset",type=int,default=10)
parser.add_argument("-p","--data_dir",type=str,default='./datasets')
parser.add_argument("--source_file", type=str )
parser.add_argument("--target_data_file", type=str)
parser.add_argument("--target_label_file", type=str)
args, _ = parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
TEST_CLASS_NUM = args.test_class_num
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class
TEST_QUERIES_NUM_PER_CLASS=args.test_queries_num_per_class




utils.same_seeds(0)


DATA_DIR = args.data_dir
DATASET_TAG = 'IP'

def _init_():
    for d in ['checkpoints']:
        if not os.path.exists(d):
            os.makedirs(d)
_init_()


with open(os.path.join(DATA_DIR, args.source_file), 'rb') as handle:
    source_imdb = pickle.load(handle)
source_imdb['data']=np.array(source_imdb['data'])
source_imdb['Labels']=np.array(source_imdb['Labels'],dtype='int')
source_imdb['set']=np.array(source_imdb['set'],dtype='int')



data_train = source_imdb['data']
labels_train = source_imdb['Labels']
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i


train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))

data = utils.sanity_check(data)
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))
        data[class_][i] = image_transpose


metatrain_data = data

del data


print(np.array(source_imdb['data']).shape)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print(source_imdb['data'].shape)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb


test_data = os.path.join(DATA_DIR, args.target_data_file)
test_label = os.path.join(DATA_DIR, args.target_label_file)

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)


def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)

    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)


    train = {}
    test = {}
    da_train = {}
    m = int(np.max(G))
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))
    print('the number of test_indices:', len(test_indices))
    print('the number of train_indices after data argumentation:', len(da_train_indices))
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb


    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth,\
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)
    train_datas, train_labels = next(train_loader.__iter__())
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth


    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)


    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())


    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}



class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class SpectralChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SpectralChannelAttention, self).__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SEBlock1d(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock1d, self).__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = x.mean(dim=2)
        w = self.fc(w).unsqueeze(2)
        return x * w


class SpectralBranch(nn.Module):
    def __init__(self, in_bands, hidden=64, out_dim=128):
        super(SpectralBranch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock1d(hidden, reduction=8)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        B, C, H, W = x.shape


        spec = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
        spec = self.conv1(spec)
        spec = self.conv2(spec)
        spec = self.se(spec)
        spec = spec.mean(dim=2)
        spec = spec.view(B, H * W, -1).mean(dim=1)
        desc = self.proj(spec)
        return desc


class SSTE(nn.Module):
    def __init__(self, in_channel, out_channel1=64, out_channel2=128,
                 patch_size=9, num_heads=4, depth=2, mlp_ratio=2.0, dropout=0.1):
        super(SSTE, self).__init__()


        self.spectral_gate = SpectralChannelAttention(in_channel, reduction=8)
        self.feature_gate = SpectralChannelAttention(out_channel2, reduction=8)


        self.spectral_branch = SpectralBranch(in_channel, hidden=64, out_dim=out_channel2)
        self.film = nn.Linear(out_channel2, out_channel2 * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel1, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channel1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channel1),
            nn.ReLU(inplace=True),
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel1, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm2d(out_channel1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel2),
            nn.ReLU(inplace=True),
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel2),
        )


        out_size = patch_size - 4
        num_tokens = out_size * out_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, out_channel2))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(out_channel2, num_heads=num_heads,
                                     mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.final_feat_dim = FEATURE_DIM

    def forward(self, x):

        x = self.spectral_gate(x)


        spec_desc = self.spectral_branch(x)


        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = F.relu(x2 + self.shortcut1(x), inplace=True)
        x3 = self.conv3(x2)
        x3 = F.relu(x3 + self.shortcut2(x2), inplace=True)
        x3 = self.feature_gate(x3)


        gamma_beta = self.film(spec_desc)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x3 = x3 * (1.0 + gamma) + beta


        B, C, H, W = x3.shape
        tokens = x3.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed


        for blk in self.blocks:
            tokens = blk(tokens)


        out = tokens.transpose(1, 2).reshape(B, C, H, W)
        return out
class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)
    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelGate, self).__init__()
        hidden = max(channels // reduction, 8)
        self.shared = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
        )
        self.gate_q = nn.Sequential(nn.Linear(hidden, channels), nn.Sigmoid())
        self.gate_s = nn.Sequential(nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, desc_q, desc_s):
        h = self.shared(torch.cat([desc_q, desc_s], dim=-1))
        return self.gate_q(h), self.gate_s(h)


class APIM(nn.Module):
    def __init__(self, dim, num_heads=4, reduction=4, temperature=1.0, attn_dropout=0.1):
        super(APIM, self).__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = temperature

        self.norm_s = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.gate = ChannelGate(dim, reduction=reduction)


        self.res_scale_q = nn.Parameter(torch.tensor(0.1))
        self.res_scale_s = nn.Parameter(torch.tensor(0.1))

    def _cross_attend(self, tokens_a, tokens_b):
        B, way, La, C = tokens_a.shape
        Lb = tokens_b.shape[2]
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(tokens_a).view(B, way, La, h, d).permute(0, 1, 3, 2, 4)
        k = self.k_proj(tokens_b).view(B, way, Lb, h, d).permute(0, 1, 3, 2, 4)
        v = self.v_proj(tokens_b).view(B, way, Lb, h, d).permute(0, 1, 3, 2, 4)

        attn = torch.matmul(q, k.transpose(-2, -1)) / ((d ** 0.5) * self.temperature)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, way, La, C)
        return out

    def forward(self, spt, qry):
        way, C, H, W = spt.shape
        num_qry = qry.shape[0]
        L = H * W

        spt_tok = self.norm_s(spt.flatten(2).transpose(1, 2))
        qry_tok = self.norm_q(qry.flatten(2).transpose(1, 2))


        spt_b = spt_tok.unsqueeze(0).expand(num_qry, way, L, C)
        qry_b = qry_tok.unsqueeze(1).expand(num_qry, way, L, C)


        qry_attn_out = self._cross_attend(qry_b, spt_b)
        spt_attn_out = self._cross_attend(spt_b, qry_b)


        gate_q, gate_s = self.gate(qry_attn_out.mean(dim=2), spt_attn_out.mean(dim=2))


        qry_out = qry_b + self.res_scale_q * (qry_attn_out * gate_q.unsqueeze(2))
        spt_out = spt_b + self.res_scale_s * (spt_attn_out * gate_s.unsqueeze(2))


        qry_pooled = qry_out.mean(dim=2)
        spt_pooled = spt_out.mean(dim=2)

        qry_embed = qry_pooled.mean(dim=1)
        spt_embed = spt_pooled.mean(dim=0)
        return spt_embed, qry_embed


class SAPNet(nn.Module):
    def __init__(self):
        super(SAPNet, self).__init__()

        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.feature_encoder = SSTE(in_channel=N_DIMENSION, out_channel1=64, out_channel2=FEATURE_DIM,
                                     patch_size=9, num_heads=4, depth=2, mlp_ratio=2.0, dropout=0.1)
        self.apim = APIM(dim=FEATURE_DIM, num_heads=4, reduction=4, temperature=1.0)

    def forward(self, spt, qry, domain='source'):

        if domain == 'target':
            spt = self.target_mapping(spt)
            qry = self.target_mapping(qry)
        elif domain == 'source':
            spt = self.source_mapping(spt)
            qry = self.source_mapping(qry)

        spt_mid = self.feature_encoder(spt)
        qry_mid = self.feature_encoder(qry)
        spt_attended, qry_attended = self.apim(spt_mid, qry_mid)

        return spt_attended, qry_attended

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def PATR(train_features, support_proto, test_features_all, class_num, shot_num_per_class,
          tau=15.0, alpha=0.1, lam=0.1, n_iters=20, move_tol=1e-4, mass_smoothing=1.0):
    with torch.no_grad():
        n_query = test_features_all.shape[0]
        n_support = train_features.shape[0]
        K = class_num
        support_labels_t = torch.arange(K, device=support_proto.device).repeat_interleave(shot_num_per_class)
        y_onehot = F.one_hot(support_labels_t, K).float()

        w = support_proto.clone()
        pi = torch.full((K,), 1.0 / K, device=w.device)

        for _ in range(n_iters):
            p_support = F.softmax(-0.5 * tau * torch.cdist(train_features, w, p=2) ** 2, dim=1)
            p_query = F.softmax(-0.5 * tau * torch.cdist(test_features_all, w, p=2) ** 2, dim=1)

            num = (p_query.clamp_min(1e-12)) ** (1.0 + alpha) * pi.sqrt().unsqueeze(0)
            denom = (p_query.clamp_min(1e-12) ** (1.0 + alpha)).sum(dim=0, keepdim=True).sqrt()
            q = num / (denom + 1e-12)
            q = q / (q.sum(dim=1, keepdim=True) + 1e-12)

            q_hat = q.mean(dim=0)
            pi = (q_hat * n_query + mass_smoothing) / (n_query + K * mass_smoothing)

            support_term = (lam / (1.0 + alpha)) * (y_onehot.t() @ train_features
                            + (p_support.sum(dim=0).unsqueeze(1) * w) - (p_support.t() @ train_features))
            query_term = (n_support / n_query) * (q.t() @ test_features_all
                          + (p_query.sum(dim=0).unsqueeze(1) * w) - (p_query.t() @ test_features_all))
            support_denom = (lam / (1.0 + alpha)) * y_onehot.sum(dim=0).unsqueeze(1)
            query_denom = (n_support / n_query) * q.sum(dim=0).unsqueeze(1)
            new_w = (support_term + query_term) / (support_denom + query_denom + 1e-8)

            move = (new_w - w).norm(dim=1).max().item()
            w = new_w
            if move < move_tol:
                break

        base_support_acc = (euclidean_metric(train_features, support_proto).argmax(1) == support_labels_t).float().mean()
        refined_support_acc = (euclidean_metric(train_features, w).argmax(1) == support_labels_t).float().mean()
        if refined_support_acc < base_support_acc:
            w = support_proto.clone()
        return w


nDataSet = args.n_dataset
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TEST_CLASS_NUM])
P = np.zeros([nDataSet, TEST_CLASS_NUM])
k = np.zeros([nDataSet, 1])
training_time = np.zeros([nDataSet, 1])
test_time = np.zeros([nDataSet, 1])
best_acc_all = 0.0
best_iDataSet = 0
seeds = [1337, 1220, 1336, 1330, 1224, 1236, 1226, 1235, 1233, 1229]
for iDataSet in range(nDataSet):

    np.random.seed(seeds[iDataSet % len(seeds)])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)

    feature_encoder = SAPNet()
    print(get_parameter_number(feature_encoder))

    feature_encoder.apply(weights_init)

    feature_encoder.cuda()

    feature_encoder.train()

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate, weight_decay=2e-4)

    DECAY_EPISODES = 4000
    def _lr_lambda(ep):
        progress = min(1.0, ep / float(DECAY_EPISODES))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    feature_encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(feature_encoder_optim, _lr_lambda)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []
    no_improve_rounds = 0
    EARLY_STOP_PATIENCE = 10

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(EPISODE):

        try:
            source_data, source_label = next(source_iter)
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = next(source_iter)

        try:
            target_data, target_label = next(target_iter)
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = next(target_iter)


        if episode % 2 == 0:

            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)


            supports, support_labels = next(support_dataloader.__iter__())
            querys, query_labels = next(query_dataloader.__iter__())


            support_features, query_features = feature_encoder(supports.cuda(), querys.cuda())


            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            else:
                support_proto = support_features


            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())
            loss = f_loss


            feature_encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        else:

            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)


            supports, support_labels = next(support_dataloader.__iter__())
            querys, query_labels = next(query_dataloader.__iter__())


            support_features, query_features = feature_encoder(supports.cuda(), querys.cuda(), domain='target')


            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            else:
                support_proto = support_features


            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())

            loss = f_loss


            feature_encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        feature_encoder_scheduler.step()

        if (episode + 1) % 100 == 0:
            elapsed_time = time.time()-train_start
            train_loss.append(loss.item())
            print('episode {:>3d}: loss: {:6.4f}, query_sample_num: {:>3d}, acc {:6.4f}, elapsed time: {:6.4f}'.format(episode + 1,\
                                                                                                                loss.item(),
                                                                                                                querys.shape[0],
                                                                                                                total_hit / total_num,
                                                                                                                   elapsed_time))
        _test_interval = 500
        if (episode + 1) % _test_interval == 0:

            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            labels_al = np.array([], dtype=np.int64)

            with torch.no_grad():
                train_datas, train_labels = next(train_loader.__iter__())
                test_datas, _ = next(test_loader.__iter__())
                train_features, _ = feature_encoder(Variable(train_datas).cuda(), Variable(test_datas).cuda(), domain='target')

                base_mean = train_features.mean(dim=0, keepdim=True)

                def _cl2n(x):
                    x = x - base_mean
                    return x / (x.norm(dim=1, keepdim=True) + 1e-8)

                train_features = _cl2n(train_features)

                support_proto=train_features.reshape(-1,TEST_LSAMPLE_NUM_PER_CLASS,FEATURE_DIM).permute(1,0,2)
                support_proto=support_proto.mean(0)
                support_proto=support_proto/support_proto.norm(dim=1, keepdim=True)

                features_pseudo=np.zeros((1,FEATURE_DIM))
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    _, test_features = feature_encoder(Variable(train_datas).cuda(), Variable(test_datas).cuda(), domain='target')
                    test_features = _cl2n(test_features)

                    test_labels = test_labels.numpy()
                    labels_al = np.append(labels_al, test_labels)
                    features_pseudo=np.append(features_pseudo,test_features.detach().cpu(),axis=0)
                    counter += batch_size

                features_pseudo=features_pseudo[1:,:]
                print("size of test features",features_pseudo.shape)

                test_features_all = torch.tensor(features_pseudo, dtype=torch.float32).cuda()

                rectified_protos = PATR(train_features, support_proto, test_features_all,
                                          TEST_CLASS_NUM, TEST_LSAMPLE_NUM_PER_CLASS)

                dist = euclidean_metric(test_features_all, rectified_protos)
                predict_labels = torch.argmax(dist, dim=1).cpu()
                print("shape is",predict_labels.shape)

                del test_features_all, dist, rectified_protos, train_features, support_proto

            rewards = [1 if predict_labels[j] == labels_al[j] else 0 for j in range(counter)]

            total_rewards = np.sum(rewards)

            predict = predict_labels
            accuracy = total_rewards / 1.0 / counter
            accuracies.append(accuracy)
            print(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()


            feature_encoder.train()
            torch.cuda.empty_cache()
            if test_accuracy > last_accuracy:

                torch.save(feature_encoder.state_dict(),str( "checkpoints/SAPFSL_feature_encoder_" + DATASET_TAG + "_" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode
                no_improve_rounds = 0

                acc[iDataSet] = total_rewards / len(test_loader.dataset)
                OA = acc[iDataSet]
                C = metrics.confusion_matrix(labels_al, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
                P[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                k[iDataSet] = metrics.cohen_kappa_score(labels_al, predict)
            else:
                no_improve_rounds += 1

            
            if no_improve_rounds >= EARLY_STOP_PATIENCE:
                print('early stopping: no improvement for {} test rounds'.format(EARLY_STOP_PATIENCE))
                break
    training_time[iDataSet] = train_end - train_start
    test_time[iDataSet] = test_end - train_end

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_acc_all = test_accuracy
        best_iDataSet = iDataSet
    print('***********************************************************************************')


classes_num = TEST_CLASS_NUM
ITER = nDataSet


print('\n================= Final Test Results =================')
for _iDataSet in range(nDataSet):
    print('--- Run {}/{} ---'.format(_iDataSet + 1, nDataSet))
    for _c in range(classes_num):
        print('  Class {:>2d} accuracy: {:6.2f}%'.format(_c + 1, A[_iDataSet, _c] * 100))
    print('  AA   : {:.2f}%'.format(np.mean(A[_iDataSet]) * 100))
    print('  OA   : {:.2f}%'.format(acc[_iDataSet, 0] * 100))
    print('  Kappa: {:.2f}%'.format(k[_iDataSet, 0] * 100))
if nDataSet > 1:
    print('--------------------------------------------------------')
    print('Mean OA    over {} run(s): {:.2f}% +/- {:.2f}%'.format(nDataSet, np.mean(acc) * 100, np.std(acc) * 100))
    print('Mean AA    over {} run(s): {:.2f}% +/- {:.2f}%'.format(nDataSet, np.mean(A) * 100, np.std(np.mean(A, axis=1)) * 100))
    print('Mean Kappa over {} run(s): {:.2f}% +/- {:.2f}%'.format(nDataSet, np.mean(k) * 100, np.std(k) * 100))
print('=========================================================\n')



print('Best of {} run(s): run {} with OA={:.2f}%'.format(nDataSet, best_iDataSet + 1, best_acc_all * 100))

best_run_checkpoint = "checkpoints/SAPFSL_feature_encoder_" + DATASET_TAG + "_" + str(best_iDataSet) + "iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"
final_best_checkpoint = "checkpoints/SAPFSL_{}_FINAL_BEST_{}shot.pkl".format(DATASET_TAG, TEST_LSAMPLE_NUM_PER_CLASS)
shutil.copyfile(best_run_checkpoint, final_best_checkpoint)
print('Final best model saved to', final_best_checkpoint)


