import pdb

import torch
import torch.nn.functional as F
from torch.nn import Linear


def create_model(in_channels, out_channels, model_args):
    if model_args['arch'] == 'GCN':
        model = GCN(in_channels, model_args['hidden_channels'], out_channels, model_args['num_layers'],
                    model_args['dropout'], model_args['activation'])
    elif model_args['arch'] == 'GCN2':
        model = GCN2(in_channels, model_args['hidden_channels'], out_channels, model_args['num_layers'],
                    model_args['dropout'], model_args['activation'])
    elif model_args['arch'] == 'GraphSAGE':
        model = GraphSAGE(in_channels, model_args['hidden_channels'], out_channels, model_args['num_layers'],
                          model_args['dropout'], model_args['activation'])
    elif model_args['arch'] == 'MLP':
        model = MLP(in_channels, model_args['hidden_channels'], out_channels, model_args['num_layers'],
                    model_args['dropout'], model_args['activation'])
    else:
        raise NotImplementedError
    return model


def get_activation(activation):
    if activation.lower() == 'relu':
        return F.relu
    elif activation.lower() == 'tanh':
        return torch.tanh
    else:
        raise NotImplementedError


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, activation):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.linears = torch.nn.ModuleList()

        if num_layers == 1:
            self.linears.append(Linear(in_channels, out_channels))
        else:
            self.linears.append(Linear(in_channels, hidden_channels))
        for _ in range(max(0, num_layers - 2)):
            self.linears.append(Linear(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.linears.append(Linear(hidden_channels, out_channels))

        for lin in self.linears:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

        self.act = get_activation(activation)

    def forward(self, x, adj, root_subgraph=None):
        for i in range(self.num_layers):
            x = self.linears[i](x)

            if i < self.num_layers - 1:
                x = self.act(x)
                # x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCNConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Linear(in_channels, out_channels, bias=bias)
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)

    def forward(self, x, adj_norm):
        x = self._spmm(x, adj_norm)
        out = self.lin(x)

        return out

    def _spmm(self, x, adj_norm):
        return torch.sparse.mm(adj_norm, x)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, activation):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

        self.act = get_activation(activation)

    def forward(self, x, adj, roots=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            x = self.act(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        x = F.normalize(x)
        return x


class GCNConv2(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Linear(in_channels, out_channels, bias=bias)
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)

    def forward(self, x, adj_norm):
        x = self.lin(x)
        out = self._spmm(x, adj_norm)

        return out

    def _spmm(self, x, adj_norm):
        return torch.sparse.mm(adj_norm, x)


class GCN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, activation):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv2(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv2(hidden_channels, hidden_channels))
        self.linear = Linear(hidden_channels, out_channels)

        self.act = get_activation(activation)

        self.dropout = dropout

    def forward(self, x, adj, roots=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            x = self.act(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        x = F.normalize(x)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, activation):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.aggregators = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        self.aggregators.append(Linear(in_channels, hidden_channels, bias=True))
        for _ in range(num_layers - 1):
            self.aggregators.append(Linear(hidden_channels, hidden_channels, bias=True))

        if num_layers == 1:
            self.linears.append(Linear(in_channels + hidden_channels, out_channels, bias=True))
        else:
            self.linears.append(Linear(in_channels + hidden_channels, hidden_channels, bias=True))
        for _ in range(max(0, num_layers - 2)):
            self.linears.append(Linear(hidden_channels * 2, hidden_channels, bias=True))
        if num_layers > 1:
            self.linears.append(Linear(hidden_channels * 2, out_channels, bias=True))

        for lin in self.linears:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
        for lin in self.aggregators:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

        self.act = get_activation(activation)

    def forward(self, x, adj, roots=None):
        for i in range(self.num_layers):
            agg = self.aggregators[i](self._spmm(x, adj))
            agg = self.act(agg)

            x = self.linears[i](torch.concat((x, agg), 1))

            if i < self.num_layers - 1:
                x = self.act(x)
                # x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.normalize(x)
        return x

    def _spmm(self, x, adj_norm):
        return torch.sparse.mm(adj_norm, x)
