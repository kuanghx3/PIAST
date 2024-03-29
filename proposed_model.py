import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from tqdm import tqdm
from args import dev
import myfunctions as fn
import pandas as pd
from torch.autograd import grad


def jacobian(inputs, outputs):
    return torch.stack([grad([outputs[:, i].sum()], [inputs], retain_graph=True, create_graph=False)[0] for i in range(outputs.size(1))], dim=-1)


class MultiHeadsGATLayer(nn.Module):
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout=0, alpha=0.2):  # input_dim = seq_length
        super(MultiHeadsGATLayer, self).__init__()

        self.head_n = head_n
        self.heads_dict = dict()
        for n in range(head_n):
            self.heads_dict[n, 0] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=dev))
            self.heads_dict[n, 1] = nn.Parameter(torch.zeros(size=(1, 2 * out_dim), device=dev))
            nn.init.xavier_normal_(self.heads_dict[n, 0], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 1], gain=1.414)
        self.linear = nn.Linear(head_n, 1)

        # regularization
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)

        # sparse metrics
        self.a_sparse = a_sparse
        self.edges = a_sparse.indices()
        self.values = a_sparse.values()
        self.N = a_sparse.shape[0]
        a_dense = a_sparse.to_dense()
        a_dense[torch.where(a_dense == 0)] = -1000000000
        a_dense[torch.where(a_dense == 1)] = 0
        self.mask = a_dense

    def forward(self, x):
        b, n, s = x.shape
        x = x.reshape(b*n, s)

        atts_stack = []
        # multi-heads attention
        for n in range(self.head_n):
            h = torch.matmul(x, self.heads_dict[n, 0])
            edge_h = torch.cat((h[self.edges[0, :], :], h[self.edges[1, :], :]), dim=1).t()  # [Ni, Nj]
            atts = self.heads_dict[n, 1].mm(edge_h).squeeze()
            atts = self.leakyrelu(atts)
            atts_stack.append(atts)

        mt_atts = torch.stack(atts_stack, dim=1)
        mt_atts = self.linear(mt_atts)
        new_values = self.values * mt_atts.squeeze()
        atts_mat = torch.sparse_coo_tensor(self.edges, new_values)
        atts_mat = atts_mat.to_dense() + self.mask
        atts_mat = self.softmax(atts_mat)
        return atts_mat

class Temporal(nn.Module):
    def __init__(self, args):
        super(Temporal, self).__init__()
        self.nodes = args.nodes
        self.seq_len = args.LOOK_BACK - args.kcnn + 1
        self.layer = args.layer
        self.LSTMlong = nn.LSTM(input_size=self.layer, hidden_size=self.layer, num_layers=2, batch_first=True)
        self.Q_linear = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.K_linear = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.V_linear = nn.Linear(self.seq_len, self.seq_len, bias=False)

    def forward(self, x):

        x_long = x
        x_long, _ = self.LSTMlong(x_long)
        y_long = x_long[:,-1,0]
        x_long = x_long.permute(0, 2, 1)
        # --------ATTEN-----------
        Q = self.Q_linear(x_long)
        K = self.K_linear(x_long).permute(0, 2, 1)
        V = self.V_linear(x_long)
        alpha = torch.matmul(Q, K) / K.shape[2]
        alpha = F.softmax(alpha, dim=2)
        ATTEN_out = torch.matmul(alpha, V)
        return ATTEN_out, y_long

class proposed_Model(nn.Module):
    def __init__(self, args, a_sparse):
        super(proposed_Model, self).__init__()
        self.hidden_size = args.nodes
        self.input_size = args.nodes
        self.output_size = args.nodes
        self.seq_len = args.LOOK_BACK
        self.output_size = args.nodes
        self.alpha = args.alpha
        self.kcnn = args.kcnn
        self.seq = self.seq_len - self.kcnn + 1
        self.conv2d = nn.Conv2d(1, 1, (self.kcnn, 2))
        self.gat_lyr = MultiHeadsGATLayer(a_sparse, self.seq, self.seq, head_n=4)
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)
        self.Temporal = Temporal(args)
        self.MLP_decoder1 = torch.nn.Sequential(
            torch.nn.Linear(self.seq, int(args.MLP_hidden)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(args.MLP_hidden), int(args.MLP_hidden / 2)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(args.MLP_hidden / 2), 1))
        self.MLP_decoder2 = torch.nn.Sequential(
            torch.nn.Linear((args.layer+1), 8*(args.layer+1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8*(args.layer+1), 4*(args.layer+1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4*(args.layer+1), 1))
        self.dropout = nn.Dropout(p=0.2)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, occ, prc_l):
        b, s, n = occ.shape
        occ = occ.permute(0, 2, 1)
        prc = prc_l.unsqueeze(2)
        prc = prc.repeat(1,1,12)
        fea = torch.stack([occ, prc],dim=3).reshape(b*n, s, -1).unsqueeze(1)
        fea = self.conv2d(fea)
        fea = fea.squeeze().reshape(b, n, self.seq)

        # first layer
        atts_mat = self.gat_lyr(fea)  # 注意力矩阵 dense(nodes, nodes)
        occ_conv1 = torch.matmul(atts_mat, fea)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        # second layer
        atts_mat = self.gat_lyr(occ_conv1)  # 注意力矩阵 dense(nodes, nodes)
        occ_conv2 = torch.matmul(atts_mat, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))
        occ_conv = torch.stack([fea, occ_conv1, occ_conv2],dim=3)
        occ_conv = occ_conv.reshape(b*n, self.seq, -1)
        ATTEN_out, y_long = self.Temporal(occ_conv)
        ATTEN_out = self.MLP_decoder1(ATTEN_out).squeeze(2)
        y_long = y_long.unsqueeze(1)
        temp_out = torch.cat([y_long, ATTEN_out], dim=1)
        y = self.MLP_decoder2(temp_out).squeeze(1)
        y = y.reshape(b,n)
        return y

class GraphConvolution2(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution2, self).__init__()
        self.linear = nn.Linear(input_size, output_size)


    def forward(self, adj, features):
        features = features.float()
        adj = adj.float()
        out = torch.matmul(adj, features)
        out = self.linear(out)
        return out


class GCN3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN3, self).__init__()
        self.gcn1 = GraphConvolution2(input_size, input_size)
        self.decoder = nn.Linear(in_features=input_size, out_features=1)

    def forward(self, adj1, features):
        out = self.gcn1(adj1, features)
        return out


class TGCN(nn.Module) :
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of GRU to stack
    """
    def __init__(self,args, a_sparse):
        super().__init__()
        adj = pd.read_csv(args.data_path + 'SZ247_adj.csv', index_col=0, header=0)
        adj_dense = np.array(adj, dtype=float)
        adj_dense = torch.Tensor(adj_dense).to(dev)
        self.adj1 = adj_dense
        self.gru = nn.GRU(input_size=1, hidden_size=1, num_layers=2, batch_first=True)
        self.hidden_size = args.nodes
        self.input_size = args.nodes
        self.output_size = args.nodes
        self.seq_len = args.LOOK_BACK
        self.output_size = args.nodes
        self.alpha = args.alpha
        self.kcnn = args.kcnn
        self.seq = self.seq_len - self.kcnn + 1
        self.conv2d = nn.Conv2d(1, 1, (self.kcnn, 2))
        self.gcn = GCN3(self.seq, self.seq, self.seq)
        self.fc = nn.Linear(in_features=self.seq, out_features=1)

    def forward(self, occ, prc_l):
        b, s, n = occ.shape
        occ = occ.permute(0, 2, 1)
        prc = prc_l.unsqueeze(2)
        prc = prc.repeat(1, 1, 12)
        fea = torch.stack([occ, prc], dim=3).reshape(b * n, s, -1).unsqueeze(1)
        fea = self.conv2d(fea)
        fea = fea.squeeze().reshape(b, n, self.seq)
        x = self.gcn(self.adj1, fea)
        x = (x+fea)/2
        x = x.reshape(b*n, self.seq, 1)
        x, _ = self.gru(x)
        x = x.squeeze(2)
        x = x.reshape(b, n, self.seq)
        x = self.fc(x)
        x = x.squeeze(2)
        return x

class GGRU(nn.Module) :
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of GRU to stack
    """
    def __init__(self,args, a_sparse):
        super().__init__()
        adj = pd.read_csv(args.data_path + 'SZ247_adj.csv', index_col=0, header=0)
        adj_dense = np.array(adj, dtype=float)
        adj_dense = torch.Tensor(adj_dense).to(dev)
        self.adj1 = adj_dense
        self.gcn = GCN3(1, 1, 1)
        self.hidden_size = args.nodes
        self.input_size = args.nodes
        self.output_size = args.nodes
        self.seq_len = args.LOOK_BACK
        self.output_size = args.nodes
        self.alpha = args.alpha
        self.kcnn = args.kcnn
        self.seq = self.seq_len - self.kcnn + 1
        self.conv2d = nn.Conv2d(1, 1, (self.kcnn, 2))
        self.grucell = nn.GRUCell(input_size=1,hidden_size=1)
        self.fc = nn.Linear(in_features=self.seq, out_features=1)

    def forward(self, occ, prc_l):
        b, s, n = occ.shape
        occ = occ.permute(0, 2, 1)
        prc = prc_l.unsqueeze(2)
        prc = prc.repeat(1, 1, 12)
        fea = torch.stack([occ, prc], dim=3).reshape(b * n, s, -1).unsqueeze(1)
        fea = self.conv2d(fea)
        x = fea.squeeze().reshape(b, n, self.seq)
        hid = None
        if hid is None:
            hid = torch.randn(b*n,1).to(dev)
        for i in range(self.seq):
            x_in = x[:, :, i].unsqueeze(2)
            x_in = self.gcn(self.adj1, x_in)
            x_in = x_in.reshape(b * n, 1)
            hid = self.grucell(x_in,hid)
        hid = hid.reshape(b,n)
        y = hid
        return y

class PhysicsInformedNN():
    def __init__(self, args, a_sparse):
        self.nodes = args.nodes
        self.p_nodes = args.p_nodes
        self.p_data_path = args.p_data_path
        self.model = proposed_Model(args, a_sparse).to(dev)
        self.lambda_1 = torch.rand(self.nodes).to(dev)*(-1)
        self.lambda_1.requires_grad_()
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.model.register_parameter('lambda_1', self.lambda_1)

    def net_y(self, occ, prc_l):
        prc_l.requires_grad_()
        occ.requires_grad_()
        y = self.model(occ, prc_l)
        return y

    def net_con1(self, occ, prc_l):
        lambda_1 = self.lambda_1
        prc_l.requires_grad_()
        occ.requires_grad_()
        y = self.net_y(occ, prc_l)
        y_p_j = jacobian(prc_l, y)
        # eye = torch.eye(self.nodes).to(dev)
        y_p = torch.diagonal(y_p_j, dim1=-2, dim2=-1)
        y_2 = y.unsqueeze(2).repeat(1, 1, self.nodes)
        Q = torch.diagonal(y_2, dim1=-2, dim2=-1)
        prc_l_2 = prc_l.unsqueeze(2).repeat(1, 1, self.nodes)
        P = torch.diagonal(prc_l_2, dim1=-2, dim2=-1)
        con1 = y_p - Q * (1.0 / P) * lambda_1
        return con1

    def net_con2(self, occ, prc_l):
        lambda_1 = self.lambda_1
        prc_l.requires_grad_()
        occ.requires_grad_()
        y = self.net_y(occ, prc_l)
        y_p = torch.autograd.grad(
            y, prc_l,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=False
        )[0]
        con1 = y_p - y * (1.0 / prc_l) * lambda_1
        con2 = lambda_1
        return con1, con2

    def train(self, train_loader, args):
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_function = torch.nn.MSELoss()
        loss_output = []
        self.model.train()
        train_losses = []
        for epoch in tqdm(range(args.max_epochs_1)):
            for t, data in enumerate(train_loader):
                occ, label, prc_l = data
                optimizer.zero_grad()
                predict = self.net_y(occ, prc_l)
                con1, con2 = self.net_con2(occ, prc_l)
                # loss = args.alpha2 * loss_function(predict, label) + (1 - args.alpha2) * torch.mean(con1 ** 2)
                L1 = loss_function(predict, label)
                L2 = torch.mean(con1 ** 2)
                L3 = torch.mean((con2 + args.e_begin) ** 2)
                loss = L1+L2 + 1*L3
                # loss = loss_function(predict, label)
                train_losses.append((loss.item()))
                loss.backward()
                optimizer.step()
                self.model.lambda_1.data.clamp_(-1, 0)
            train_loss = np.average(train_losses)
            loss_output.append(train_loss)
            print_msg = f'train_loss: {train_loss:.8f} '
            if (epoch + 1) % 100 == 0:
                print(print_msg)
                print(self.model.lambda_1.detach().cpu().numpy())

        for epoch in tqdm(range(args.max_epochs_2)):
            for t, data in enumerate(train_loader):
                occ, label, prc_l = data
                optimizer.zero_grad()
                predict = self.net_y(occ, prc_l)
                con1,con2 = self.net_con2(occ, prc_l)
                #loss = args.alpha2 * loss_function(predict, label) + (1 - args.alpha2) * torch.mean(con1 ** 2)
                L1 = loss_function(predict, label)
                L2 = torch.mean(con1 ** 2)
                L3 = torch.mean((con2+args.e_begin)**2)
                loss = L1+L2+0.1*L3
                # loss = loss_function(predict, label)
                train_losses.append((loss.item()))
                loss.backward()
                optimizer.step()
                self.model.lambda_1.data.clamp_(-1, 0)
            train_loss = np.average(train_losses)
            loss_output.append(train_loss)
            print_msg = f'train_loss: {train_loss:.8f} '
            if (epoch + 1) % 100 == 0:
                print(print_msg)
                print(self.model.lambda_1.detach().cpu().numpy())

        for epoch in tqdm(range(args.max_epochs_3)):
            for t, data in enumerate(train_loader):
                occ, label, prc_l = data
                optimizer.zero_grad()
                predict = self.net_y(occ, prc_l)
                # con1, con2 = self.net_con2(occ, prc_l)
                # loss = args.alpha2 * loss_function(predict, label) + (1 - args.alpha2) * torch.mean(con1 ** 2)
                L1 = loss_function(predict, label)
                # L2 = torch.mean(con1 ** 2)
                # L3 = torch.mean((con2 + args.e_begin) ** 2)
                loss = L1
                # loss = loss_function(predict, label)
                train_losses.append((loss.item()))
                loss.backward()
                optimizer.step()
                self.model.lambda_1.data.clamp_(-1, 0)
            train_loss = np.average(train_losses)
            loss_output.append(train_loss)
            print_msg = f'train_loss: {train_loss:.8f} '
            if (epoch + 1) % 100 == 0:
                print(print_msg)
                print(self.model.lambda_1.detach().cpu().numpy())

        pd.DataFrame(data=loss_output).to_csv("./result_" + str(args.predict_time) + "/" + "loss.csv")

    def predict(self, test_loader, args):
        approach = "proposed"
        result = []
        self.model.eval()
        for t, data in enumerate(test_loader):
            occ, label, prc_l = data
            occ.requires_grad_()
            prc_l.requires_grad_()
            predict = self.net_y(occ, prc_l)
            metrics = fn.get_metrics(predict.cpu().detach().numpy(), label.cpu().detach().numpy())
            result.append(metrics)
        pd.DataFrame(data=result, columns=["MSE", "RMSE", "MAPE", "RAE", "MAE", "R2"]).to_csv(
            "./result_" + str(args.predict_time) + "/" + approach + ".csv")
        return result

class PhysicsInformedNN2():
    def __init__(self, args, a_sparse, nodes):
        # nodes = args.nodes
        self.nodes = nodes
        self.p_nodes = args.p_nodes
        self.p_data_path = args.p_data_path
        self.model = proposed_Model(args, a_sparse).to(dev)
        self.lambda_1 = torch.rand(self.nodes).to(dev)*(-1)
        self.lambda_1.requires_grad_()
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.model.register_parameter('lambda_1', self.lambda_1)

    def net_y(self, occ, prc_l):
        prc_l.requires_grad_()
        occ.requires_grad_()
        y = self.model(occ, prc_l)
        return y

    def net_con1(self, occ, prc_l):
        lambda_1 = self.lambda_1
        prc_l.requires_grad_()
        occ.requires_grad_()
        y = self.net_y(occ, prc_l)
        y_p_j = jacobian(prc_l, y)
        # eye = torch.eye(self.nodes).to(dev)
        y_p = torch.diagonal(y_p_j, dim1=-2, dim2=-1)
        y_2 = y.unsqueeze(2).repeat(1, 1, self.nodes)
        Q = torch.diagonal(y_2, dim1=-2, dim2=-1)
        prc_l_2 = prc_l.unsqueeze(2).repeat(1, 1, self.nodes)
        P = torch.diagonal(prc_l_2, dim1=-2, dim2=-1)
        con1 = y_p - Q * (1.0 / P) * lambda_1
        return con1

    def net_con2(self, occ, prc_l):
        lambda_1 = self.lambda_1
        prc_l.requires_grad_()
        occ.requires_grad_()
        y = self.net_y(occ, prc_l)
        y_p = torch.autograd.grad(
            y, prc_l,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=False
        )[0]
        con1 = y_p - y * (1.0 / prc_l) * lambda_1
        con2 = lambda_1
        return con1, con2

    def train(self, train_loader, args):
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_function = torch.nn.MSELoss()
        loss_output = []
        self.model.train()
        train_losses = []
        for epoch in tqdm(range(args.max_epochs_1)):
            for t, data in enumerate(train_loader):
                occ, label, prc_l = data
                optimizer.zero_grad()
                predict = self.net_y(occ, prc_l)
                con1, con2 = self.net_con2(occ, prc_l)
                # loss = args.alpha2 * loss_function(predict, label) + (1 - args.alpha2) * torch.mean(con1 ** 2)
                L1 = loss_function(predict, label)
                L2 = torch.mean(con1 ** 2)
                L3 = torch.mean((con2 + args.e_begin) ** 2)
                loss = L1+L2 + 1*L3
                # loss = loss_function(predict, label)
                train_losses.append((loss.item()))
                loss.backward()
                optimizer.step()
                self.model.lambda_1.data.clamp_(-1, 0)
            train_loss = np.average(train_losses)
            loss_output.append(train_loss)
            print_msg = f'train_loss: {train_loss:.8f} '
            if (epoch + 1) % 100 == 0:
                print(print_msg)


        for epoch in tqdm(range(args.max_epochs_2)):
            for t, data in enumerate(train_loader):
                occ, label, prc_l = data
                optimizer.zero_grad()
                predict = self.net_y(occ, prc_l)
                con1,con2 = self.net_con2(occ, prc_l)
                #loss = args.alpha2 * loss_function(predict, label) + (1 - args.alpha2) * torch.mean(con1 ** 2)
                L1 = loss_function(predict, label)
                L2 = torch.mean(con1 ** 2)
                L3 = torch.mean((con2+args.e_begin)**2)
                loss = L1+L2+0.1*L3
                # loss = loss_function(predict, label)
                train_losses.append((loss.item()))
                loss.backward()
                optimizer.step()
                self.model.lambda_1.data.clamp_(-1, 0)
            train_loss = np.average(train_losses)
            loss_output.append(train_loss)
            print_msg = f'train_loss: {train_loss:.8f} '
            if (epoch + 1) % 100 == 0:
                print(print_msg)


        for epoch in tqdm(range(args.max_epochs_3)):
            for t, data in enumerate(train_loader):
                occ, label, prc_l = data
                optimizer.zero_grad()
                predict = self.net_y(occ, prc_l)
                # con1, con2 = self.net_con2(occ, prc_l)
                # loss = args.alpha2 * loss_function(predict, label) + (1 - args.alpha2) * torch.mean(con1 ** 2)
                L1 = loss_function(predict, label)
                # L2 = torch.mean(con1 ** 2)
                # L3 = torch.mean((con2 + args.e_begin) ** 2)
                loss = L1
                # loss = loss_function(predict, label)
                train_losses.append((loss.item()))
                loss.backward()
                optimizer.step()
                self.model.lambda_1.data.clamp_(-1, 0)
            train_loss = np.average(train_losses)
            loss_output.append(train_loss)
            print_msg = f'train_loss: {train_loss:.8f} '
            if (epoch + 1) % 100 == 0:
                print(print_msg)


        # pd.DataFrame(data=loss_output).to_csv("./result_" + str(args.predict_time) + "/" + "loss.csv")

    def predict(self, test_loader, args):
        approach = "proposed"
        result = []
        self.model.eval()
        for t, data in enumerate(test_loader):
            occ, label, prc_l = data
            occ.requires_grad_()
            prc_l.requires_grad_()
            predict = self.net_y(occ, prc_l)
            metrics = fn.get_metrics(predict.cpu().detach().numpy(), label.cpu().detach().numpy())
            result.append(metrics)
        # pd.DataFrame(data=result, columns=["MSE", "RMSE", "MAPE", "RAE", "MAE", "R2"]).to_csv(
        #     "./result_" + str(args.predict_time) + "/" + approach + ".csv")
        return result





