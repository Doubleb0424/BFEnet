import math
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GraphConv, DenseSAGEConv, dense_diff_pool, DenseGCNConv
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling
from torch.nn import Linear, Dropout, PReLU, Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse

device = torch.device('cuda', 0)


class Gnn(torch.nn.Module):

    def __init__(self, in_features: int, bn_features: int, out_features: int, topk: int):
        super().__init__()

        self.channels = 62
        self.in_features = in_features
        self.bn_features = bn_features
        self.out_features = out_features
        self.topk = topk

        self.bnlin = Linear(in_features, bn_features)
        self.gconv = DenseGCNConv(in_features, out_features)

    def forward(self, x):
        x = x.reshape(-1, self.channels, self.in_features)
        xa = torch.tanh(self.bnlin(x))
        adj = torch.matmul(xa, xa.transpose(2, 1))
        adj = torch.softmax(adj, 2)
        amask = torch.zeros(xa.size(0), self.channels, self.channels).to(device)
        amask.fill_(0.0)
        s, t = adj.topk(self.topk, 2)
        amask.scatter_(2, t, s.fill_(1))
        adj_out = adj = adj * amask
        x = F.relu(self.gconv(x, adj))
        return x, adj_out


class BFEnet(torch.nn.Module):
    def __init__(self, out_adj=False):
        super().__init__()

        drop_rate = 0.1
        topk = 10
        self.channels = 62
        self.out_adj = out_adj

        self.conv1 = Conv2d(1, 32, (1, 5))
        self.drop1 = Dropout(drop_rate)
        self.pool1 = MaxPool2d((1, 4))
        self.gnn1 = Gnn(65 * 32, 64, 32, topk)

        self.conv2 = Conv2d(32, 64, (1, 5))
        self.drop2 = Dropout(drop_rate)
        self.pool2 = MaxPool2d((1, 4))
        self.gnn2 = Gnn(15 * 64, 64, 32, topk)

        self.conv3 = Conv2d(64, 128, (1, 5))
        self.drop3 = Dropout(drop_rate)
        self.pool3 = MaxPool2d((1, 4))
        self.gnn3 = Gnn(2 * 128, 64, 32, topk)
        self.drop4 = Dropout(drop_rate)

        self.linend = Linear(self.channels * 32 * 3, self.channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)

        x1, adj1 = self.gnn1(x)

        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)

        x2, adj2 = self.gnn2(x)

        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        x = self.pool3(x)


        x3, adj3 = self.gnn3(x)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.drop4(x)

        x = x.reshape(-1, self.channels * 32 * 3)
        x = self.linend(x)

        return x


class Model(torch.nn.Module):
    def __init__(self, num_class, out_adj=False):
        super(Model, self).__init__()
        self.model_0 = BFEnet(out_adj)
        self.model_1 = BFEnet(out_adj)
        self.model_2 = BFEnet(out_adj)
        self.model_3 = BFEnet(out_adj)
        self.model_4 = BFEnet(out_adj)
        self.model = [self.model_0, self.model_1, self.model_2, self.model_3, self.model_4]
        self.channels = 62
        self.out_adj = out_adj
        self.linend = Linear(self.channels * 5, num_class)

    def forward(self, x, batch):
        x, mask = to_dense_batch(x, batch)

        x = x.reshape(-1, 1, 5, 265)
        x_feature = []
        out_adjs = []
        for i in range(5):
            if self.out_adj:
                feature, adj = self.model[i](x[:, :, i][:, :, None])
                out_adjs.append(adj)
            else:
                feature = self.model[i](x[:, :, i][:, :, None])
            x_feature.append(feature)
        out_feature = torch.concatenate(x_feature, dim=-1)
        out = self.linend(out_feature)

        return out

