import collections

import torch
from torch.nn import Linear, Embedding
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv, SGConv

# TODO проверить что edge index  data -- отражает НЕнаправленный граф
class Net(torch.nn.Module):
    def __init__(
        self,
        dataset,
        device,
        loss_function,
        mode="unsupervised",
        conv="GCN",
            RNWE_layer=None,
        hidden_layer=64,
        out_layer=128,
        dropout=0,
        num_layers=2,
        hidden_layer_for_classifier=128,
        number_of_layers_for_classifier=2,
        heads=1,

    ):
        super(Net, self).__init__()
        self.mode = mode
        self.conv = conv
        self.num_layers = num_layers
        self.data = dataset
        self.num_features = dataset.x.shape[1]
        # print(dataset.num_features)
        self.loss_function = loss_function
        self.convs = torch.nn.ModuleList()
        self.hidden_layer = hidden_layer
        self.out_layer = out_layer
        self.dropout = dropout
        self.device = device
        self.history = []
        out_channels = self.out_layer
        self.heads = heads
        self.RNWE_layer = RNWE_layer if RNWE_layer else len(self.data.x)

        if self.mode == "unsupervised":
            if loss_function["loss var"] == "Random Walks":
                self.loss = self.lossRandomWalks
            elif loss_function["loss var"] == "Context Matrix":
                self.loss = self.lossContextMatrix
            elif loss_function["loss var"] == "Factorization":
                self.loss = self.lossFactorization
            elif loss_function["loss var"] == "Laplacian EigenMaps":
                self.loss = self.lossLaplacianEigenMaps
            elif loss_function["loss var"] == "Force2Vec":
                self.loss = self.lossTdistribution
            elif loss_function["loss var"] == "GraRep":
                self.loss = self.lossGraRep
            elif loss_function["loss var"] == "RNWE":
                self.loss = self.lossRNWE
                self.layer_rnwe = Linear(self.out_layer,self.data.num_nodes)

        elif self.mode == "supervised":

            self.num_classes = len(
                collections.Counter(self.data.y.tolist()).keys()
            )
            self.hidden_layer_for_classifier = hidden_layer_for_classifier
            self.number_of_layers_for_classifier = number_of_layers_for_classifier

            self.classifier = torch.nn.ModuleList()
            if self.number_of_layers_for_classifier==1:
                self.classifier.append(torch.nn.Linear(self.heads*out_channels, self.num_classes))
            else:
                self.classifier.append(torch.nn.Linear(self.heads*out_channels, self.hidden_layer_for_classifier))
                for i in range(1,self.number_of_layers_for_classifier-1):
                    self.classifier.append(torch.nn.Linear(self.hidden_layer_for_classifier, self.hidden_layer_for_classifier))
                self.classifier.append(torch.nn.Linear(self.hidden_layer_for_classifier, self.num_classes))


        if self.conv == "GCN":
            if self.num_layers == 1:
                self.convs.append(GCNConv(self.num_features, out_channels))
            else:
                self.convs.append(GCNConv(self.num_features, self.hidden_layer))
                for i in range(1, self.num_layers - 1):
                    self.convs.append(GCNConv(self.hidden_layer, self.hidden_layer))
                self.convs.append(GCNConv(self.hidden_layer, out_channels))
        elif self.conv == "SAGE":

            if self.num_layers == 1:
                self.convs.append(SAGEConv(self.num_features, out_channels))
            else:
                self.convs.append(SAGEConv(self.num_features, self.hidden_layer))
                for i in range(1, self.num_layers - 1):
                    self.convs.append(SAGEConv(self.hidden_layer, self.hidden_layer))
                self.convs.append(SAGEConv(self.hidden_layer, out_channels))
        elif self.conv == "GAT":
            if self.num_layers == 1:
                self.convs.append(GATConv(self.num_features, out_channels,heads=self.heads))
            else:
                self.convs.append(GATConv(self.num_features, self.hidden_layer, heads=self.heads))
                for i in range(1, self.num_layers - 1):
                    self.convs.append(GATConv(self.heads*self.hidden_layer, self.hidden_layer,heads=self.heads))
                self.convs.append(GATConv(self.heads*self.hidden_layer, out_channels,heads=self.heads))
        elif self.conv == "transductive":
            self.convs.append(Embedding(len(self.data.x), out_channels, max_norm=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.mode == "unsupervised":
            if self.loss_function["loss var"] == 'RNWE':
                return torch.sigmoid(self.layer_rnwe(x)), x
            else:
                return x

        elif self.mode == "supervised":
            for j in range(self.number_of_layers_for_classifier):
                x = self.classifier[j](x)
                x = F.relu(x)
            return x.log_softmax(dim=1)



    def inference(self, data, dp=0):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=dp, training=self.training)
        if self.mode == "unsupervised":
            if self.loss_function["loss var"] == 'RNWE':
                return torch.sigmoid(self.layer_rnwe(x)), x
            else:
                return x
        elif self.mode == "supervised":
            for j in range(self.number_of_layers_for_classifier):
                x = self.classifier[j](x)
                x = F.relu(x)
            return x.log_softmax(dim=-1)
    def tr_inference(self, data, dp=0):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.convs[0](torch.LongTensor(range(len(x))))
        if self.loss_function["loss var"] == 'RNWE':
            return torch.sigmoid(self.layer_rnwe(x)), x
        else:
            return x


    def lossRandomWalks(self, out, PosNegSamples):
        (pos_rw, neg_rw) = PosNegSamples
        pos_rw, neg_rw = pos_rw.type(torch.LongTensor).to(self.device), neg_rw.type(torch.LongTensor).to(self.device)
        # Positive loss.
        pos_loss = 0
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        h_start = out[start].view(pos_rw.size(0), 1, self.out_layer)
        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1, self.out_layer)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)

        pos_loss = -(torch.nn.LogSigmoid()(dot)).mean()  # -torch.log(torch.sigmoid(dot)).mean()

        # print('dot',dot.device)
        # Negative loss
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        h_start = out[start].view(neg_rw.size(0), 1, self.out_layer)
        h_rest = out[rest.view(-1)].view(neg_rw.size(0), -1, self.out_layer)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -(torch.nn.LogSigmoid()((-1) * dot)).mean()

        return pos_loss + neg_loss  # +0.5*lmbda*sum(sum(out*out))

    def lossContextMatrix(self, out, PosNegSamples):
        (pos_rw, neg_rw) = PosNegSamples
        pos_rw = pos_rw.to(self.device)
        neg_rw = neg_rw.to(self.device)

       # print(neg_rw)
        start, rest = neg_rw[:, 0].type(torch.LongTensor), neg_rw[:, 1:].type(torch.LongTensor).contiguous()
        indices = start != rest.view(-1)
        start = start[indices]
        h_start = out[start].view(start.shape[0], 1, self.out_layer)
        rest = rest[indices]
        # print('neg',start,rest)

        h_rest = out[rest].view(rest.shape[0], -1, self.out_layer)

        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -(torch.nn.LogSigmoid()((-1) * dot)).mean()

        # Positive loss.
        start, rest = pos_rw[:, 0].type(torch.LongTensor), pos_rw[:, 1].type(torch.LongTensor).contiguous()

        weight = pos_rw[:, 2]
        h_start = out[start].view(pos_rw.size(0), 1, self.out_layer)


        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1, self.out_layer)

        dot = ((h_start * h_rest).sum(dim=-1)).view(-1)

        if self.loss_function["Name"] == "LINE":
            pos_loss = -2 * (weight * torch.nn.LogSigmoid()(dot)).mean()

        elif self.loss_function["Name"].split("_")[0] == "VERSE" or self.loss_function["Name"] == "APP":
            pos_loss = -(weight * torch.nn.LogSigmoid()(dot)).mean()

        return pos_loss + neg_loss

    def lossGraRep(self, out, PosNegSamples):
        (pos_rw, neg_rw) = PosNegSamples
        pos_rw = pos_rw.to(self.device)
        neg_rw = neg_rw.to(self.device)

       # print(neg_rw)
        start, rest = neg_rw[:, 0].type(torch.LongTensor), neg_rw[:, 1:].type(torch.LongTensor).contiguous()
        indices = start != rest.view(-1)
        start = start[indices]
        h_start = out[start].view(start.shape[0], 1, self.out_layer)
        rest = rest[indices]
        # print('neg',start,rest)

        h_rest = out[rest].view(rest.shape[0], -1, self.out_layer)

        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        lmbda = self.loss_function["lmbda"]
        neg_loss = -lmbda*(torch.nn.LogSigmoid()((-1) * dot)).mean()/len(self.data.x)

        # Positive loss.
        start, rest = pos_rw[:, 0].type(torch.LongTensor), pos_rw[:, 1].type(torch.LongTensor).contiguous()

        weight = pos_rw[:, 2]
        h_start = out[start].view(pos_rw.size(0), 1, self.out_layer)


        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1, self.out_layer)

        dot = ((h_start * h_rest).sum(dim=-1)).view(-1)

        pos_loss = - (weight * torch.nn.LogSigmoid()( dot)).mean()

        return pos_loss + neg_loss
    def lossFactorization(self, out, S, **kwargs):
        S = S.to(self.device)
        lmbda = self.loss_function["lmbda"]
        if self.loss_function["Name"] == "GraphFactorization":
            Y_product = torch.mm(out, out.t())
            loss = torch.sum((1 - Y_product) * S) + 0.5 * lmbda * sum(sum(out * out))
        else:
            loss = 0.5 * sum(
                sum((S - torch.matmul(out, out.t())) * (S - torch.matmul(out, out.t())))) + 0.5 * lmbda * sum(
                sum(out * out)
            )
        return loss

    def lossLaplacianEigenMaps(self, out, A):
        dd = torch.device("cpu")
        # dd=torch.device('cpu')
        L = (torch.diag(sum(A)) - A).type(torch.FloatTensor).to(dd)
        out_tr = out.t().to(dd)
        loss = torch.trace(torch.matmul(torch.matmul(out_tr, L), out))
        yDy = torch.matmul(
            torch.matmul(out_tr, torch.diag(sum(A.t())).type(torch.FloatTensor).to(dd)), out
        ) - torch.diag(torch.ones(out.shape[1])).type(torch.FloatTensor).to(dd)
        loss_2 = torch.sqrt(sum(sum(yDy * yDy)))
        return loss + loss_2

    def lossTdistribution(self, out, PosNegSamples):
        eps = 10e-6
        (pos_rw, neg_rw) = PosNegSamples
        pos_rw = pos_rw.to(self.device)
        neg_rw = neg_rw.to(self.device)

        start, rest = neg_rw[:, 0].type(torch.LongTensor), neg_rw[:, 1:].type(torch.LongTensor).contiguous()
        indices = start != rest.view(-1)
        start = start[indices]
        h_start = out[start].view(start.shape[0], 1, self.out_layer)
        rest = rest.view(-1)
        rest = rest[indices]
        # print('neg',start,rest)
        h_rest = out[rest].view(rest.shape[0], -1, self.out_layer)

        # h_start=torch.nn.functional.normalize(h_start, p=2.0, dim = -1)
        # h_rest=torch.nn.functional.normalize(h_rest, p=2.0, dim = -1)
        # print(h_start,h_rest)
        t_squared = ((h_start - h_rest) * (h_start - h_rest)).mean(dim=-1).view(-1)
        neg_loss = (-torch.log((t_squared / (1 + t_squared)))).mean()

        # Positive loss.
        start, rest = pos_rw[:, 0].type(torch.LongTensor), pos_rw[:, 1].type(torch.LongTensor).contiguous()
        weight = pos_rw[:, 2]
        h_start = out[start].view(pos_rw.size(0), 1, self.out_layer)
        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1, self.out_layer)
        # h_start=torch.nn.functional.normalize(h_start, p=2.0, dim = -1)
        # h_rest=torch.nn.functional.normalize(h_rest, p=2.0, dim = -1)
        t_squared = ((h_start - h_rest) * (h_start - h_rest)).sum(dim=-1).view(-1)
        pos_loss = (torch.log(1 + t_squared)).mean()
        # print('losses',pos_loss,neg_loss)

        return pos_loss + neg_loss


    def lossRNWE(self, out_all, PosNegSamples):
        Phi, out = out_all
        lmbda = self.loss_function["lmbda"]
        gamma = self.loss_function["gamma"]
        eps = 10e-6
        (pos_rw, neg_rw) = PosNegSamples

        pos_rw = pos_rw.to(self.device)
        neg_rw = neg_rw.to(self.device)
        start, rest = neg_rw[:, 0].type(torch.LongTensor), neg_rw[:, 1:].type(torch.LongTensor).contiguous()
        #indices = start != rest.view(-1)
        #start = start[indices]
        h_start = out[start].view(start.shape[0], 1, self.out_layer)
        rest = rest.view(-1)
        rest = rest#[indices]
        h_rest = out[rest].view(neg_rw.size(0), -1, self.out_layer)
        sim = (0.5)*(1+(h_start * h_rest).sum(dim=-1)/(torch.norm(h_start)*torch.norm(h_rest)))
        rows_selected = Phi[neg_rw[:, 0]]
        # Теперь для каждой строки извлекаем элементы, используя индексы из pos_rw[:, 1:]
        Phi_extended = torch.zeros((neg_rw.shape[0], neg_rw.shape[1] - 1))
        for i in range(neg_rw.shape[1] - 1):
            Phi_extended[:, i] = rows_selected[torch.arange(rows_selected.shape[0]), neg_rw[:, i + 1]]
        ones_tensor=torch.ones(sim.shape[0],sim.shape[1])
        neg_loss = - (gamma*torch.log(ones_tensor-sim) + lmbda*torch.log(ones_tensor-Phi_extended)).sum()

        start, rest = pos_rw[:, 0].type(torch.LongTensor), pos_rw[:, 1:].type(torch.LongTensor).contiguous()
        h_start = out[start].view(pos_rw.size(0), 1, self.out_layer)
        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1, self.out_layer)
        sim = (0.5)*(1+(h_start * h_rest).sum(dim=-1)/(torch.norm(h_start)*torch.norm(h_rest)))
        rows_selected = Phi[pos_rw[:, 0]]
        # Теперь для каждой строки извлекаем элементы, используя индексы из pos_rw[:, 1:]
        Phi_extended = torch.zeros((pos_rw.shape[0], pos_rw.shape[1] - 1))
        for i in range(pos_rw.shape[1] - 1):
            Phi_extended[:, i] = rows_selected[torch.arange(rows_selected.shape[0]), pos_rw[:, i + 1]]
        pos_loss = -(gamma*torch.log(sim) + lmbda*torch.log(Phi_extended)).sum()
        return pos_loss + neg_loss

    # loss function for supervised mode
    def loss_sup(self, pred, label):
        return F.nll_loss(pred, label)
