import torch
from torch import nn

from torch_geometric.nn.pool.glob import global_mean_pool
from GNN_layers import ProcessorLayer, SmoothingLayer


class FlowGNN_original(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, geom_in_dim=2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_original, self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.processor = nn.ModuleList()

        self.decoder = nn.LazyLinear(out_dim)

        for ef, nf in zip(self.edge_feat_dims, self.num_filters):
            self.processor.append(ProcessorLayer(ef, nf, hidden_nodes))
            self.processor.append(SmoothingLayer())

    def forward(self, data):

        x, edge_index, edge_attr = data.x[:, :7], data.edge_index, data.edge_attr

        skip_info = x[:, :self.geom_in_dim]

        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred

    def loss(self, pred, inp):
        true_flow = inp.flow
        print(true_flow.shape, pred.shape)
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)


class FlowGNN_original_skipBC(nn.Module):  # напрямую прокидываютс параметры гран условий в вершины

    def __init__(self, edge_feat_dims, num_filters, skipcon_indx=[], geom_in_dim=2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_original_skipBC, self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.skipcon_indx = skipcon_indx

        self.processor = nn.ModuleList()

        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef, nf, hidden_nodes, idx=i))
            self.processor.append(SmoothingLayer(idx=i))

    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,
                                   :3], data.edge_index, data.edge_attr,  # здесь были data.bc, убрал для прямых гран.усл.

        bc = data.x[:, 3:]
        x = torch.cat([x, bc], 1)

        skip_info = x[:, :self.geom_in_dim]

        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)

                # if layer.name == 'smoothing' and layer.idx == 2:
                #     x = torch.cat([x, bc], 1)

                # if layer.name == 'smoothing' and layer.idx == 5:
                #     x = torch.cat([x, bc], 1)
                # if layer.idx%3 == 0:
                if layer.idx in self.skipcon_indx:  # if layer.idx%3 ==0
                    x = torch.cat([x, bc], 1)

        pred = self.decoder(x)
        return pred

    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)


class FlowGNN_conv_block(nn.Module):

    def __init__(self, edge_dims, node_dims, hidden_size=128, node_skip_cons_idx=None, edge_skip_cons_idx=None,
                 fc_con=False, batchnorm=False, idx=0, selu=False):
        super(FlowGNN_conv_block, self).__init__()
        self.conv = ProcessorLayer(edge_dims, node_dims, hidden_size, selu)
        self.smooth = SmoothingLayer()

        self.node_norm_layer = None
        if batchnorm:
            self.node_norm_layer = nn.BatchNorm1d(node_dims)

        self.node_skip_cons_idx = node_skip_cons_idx  # int, layer indx
        self.edge_skip_cons_idx = edge_skip_cons_idx
        self.fc_con = fc_con  # bool,
        self.idx = idx

    def forward(self, node_attr, edge_idx, edge_attr, node_skip_cons=None, edge_skip_cons=None, fc_con=None,
                skip_info=None):

        if node_skip_cons is not None:
            node_attr = torch.add(node_attr, node_skip_cons)

        if edge_skip_cons is not None:
            edge_attr = torch.add(edge_attr, edge_skip_cons)

        if skip_info is not None:
            node_attr = torch.cat([node_attr, skip_info], 1)

        if self.fc_con:
            node_attr = torch.cat([node_attr, fc_con], 1)

        node_attr, edge_attr = self.conv(node_attr, edge_idx, edge_attr)
        node_attr, edge_attr = self.smooth(node_attr, edge_idx, edge_attr)

        if self.node_norm_layer is not None:
            node_attr = self.node_norm_layer(node_attr)

        return node_attr, edge_attr


class FlowGNN_fc_block(nn.Module):

    def __init__(self, out_dim, hidden_layers):
        super(FlowGNN_fc_block, self).__init__()
        self.out_dim = out_dim
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()

        for hidden_dim in self.hidden_layers:
            self.layers.append(nn.LazyLinear(hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.LazyLinear(self.out_dim))

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            # print('FC hui')
            # print('FC layer num', i)
            x = layer(x)
        # print('FC', )
        return x


class FlowGNN(nn.Module):  # универсальная модель

    def __init__(self, edge_filters, node_filters, fc_in_dim, fc_out_dim,
                 node_skip_cons_list=None, edge_skip_cons_list=None,
                 fc_con_list=None, fc_hidden_layers=(128, 128),
                 batchnorm=None, selu=None,
                 geom_in_dim=2, out_dim=4):
        super(FlowGNN, self).__init__()

        self.edge_filters = edge_filters
        self.node_filters = node_filters
        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim

        # каждому слою сопоставляется множество слоев из которых приходят SC
        # [None, None, None, [0,1,2],...] В блок № 3 прокинуты скип коннекшены из блоков №№0-2
        self.node_skip_cons_list = node_skip_cons_list
        self.edge_skip_cons_list = edge_skip_cons_list

        self.fc_con_list = fc_con_list  # [1,3,4,6] слои куда заходит fc слой
        self.fc_hidden_layers = fc_hidden_layers

        self.batchnorm = batchnorm
        self.selu = selu
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.gcnn_layers_list = nn.ModuleList()
        self.fc_layers_list = nn.ModuleList()

        for i, (ef, nf) in enumerate(zip(self.edge_filters, self.node_filters)):

            print(ef)
            print(nf)
            fc_con = False
            if i in self.fc_con_list:
                fc_con = True
                self.fc_layers_list.append(FlowGNN_fc_block(self.fc_out_dim, self.fc_hidden_layers))
            self.gcnn_layers_list.append(FlowGNN_conv_block(ef, nf,
                                                            node_skip_cons_idx=self.node_skip_cons_list[i],
                                                            edge_skip_cons_idx=self.edge_skip_cons_list[i],
                                                            fc_con=fc_con,
                                                            batchnorm=self.batchnorm[i], idx=i,
                                                            selu=self.selu[i]))

        self.decoder = nn.LazyLinear(self.out_dim)

    def forward(self, data):
        x = data.x

        x_outs = {}  # nodes
        edge_outs = {}  # edges
        skip_con_nodes = None
        skip_con_edges = None
        skip_info = x[:, :self.geom_in_dim]
        fc_out = None
        if self.fc_con_list is not None:
            fc_out = self.fc_layers_list[0](data.bc)
        fc_count = 1

        for i, layer in enumerate(self.gcnn_layers_list):

            if layer.node_skip_cons_idx is not None:
                skip_con_nodes = x_outs[layer.node_skip_cons_idx[0]]
                for i in layer.node_skip_cons_idx[1:]:
                    skip_con_nodes += x_outs[i]

            if layer.edge_skip_cons_idx is not None:
                skip_con_edges = edge_outs[layer.edge_skip_cons_idx[0]]
                for i in layer.edge_skip_cons_idx[1:]:
                    skip_con_edges += edge_outs[i]

            if layer.idx in self.fc_con_list[1:]:
                graph_pool = global_mean_pool(x, data.batch)
                graph_pool = graph_pool[data.batch]
                fc_out = self.fc_layers_list[fc_count](torch.cat([data.bc, fc_out, graph_pool], 1))
                fc_count += 1
            x, edge_attr = layer(x, data.edge_index, data.edge_attr, skip_con_nodes, skip_con_edges, fc_out, skip_info)

            skip_con_nodes = None
            skip_con_edges = None
            x_outs[i] = x
            edge_outs[i] = edge_attr

        pred = self.decoder(x)

        return pred
