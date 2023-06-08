import pickle
import os
import sys
import joblib

sys.path.append('../')
from modules.negativeSampling import NegativeSampler
import torch
import numpy as np
import random
from torch_geometric.data import Data
from modules.model import Net
from modules.sampling import SamplerContextMatrix, SamplerRandomWalk, SamplerFactorization, SamplerAPP
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborSampler
import optuna
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

synthetic = True

benchmark_data_dir = "../dataset/"
help_data = "../data_help/"
if synthetic:

    def data_load(name):
        x = torch.tensor(np.load(f'{benchmark_data_dir}/graph_' + str(name) + '_attr.npy'), dtype=torch.float)
        edge_list = torch.tensor(np.load(f'{benchmark_data_dir}/graph_' + str(name) + '_edgelist.npy')).t()

        data = Data(x=x, edge_index=edge_list)
        indices = list(range(len(data.x)))

        train_indices = random.sample(list(range(len(indices))), k=int(len(indices) * 0.7))
        train_indices_torch = torch.tensor(train_indices)

        test_indices = list((set(indices) - set(train_indices)))
        test_indices_torch = torch.tensor(test_indices)
        train_mask = torch.tensor([False] * len(indices))
        test_mask = torch.tensor([False] * len(indices))
        train_mask[train_indices] = True
        test_mask[test_indices] = True

        return data, train_indices_torch, test_indices_torch, train_mask, test_mask
else:

    def data_load(name):
        if name == 'Cora' or name == 'Citeseer' or name == 'Pubmed':
            data = Planetoid(root='/tmp/' + str(name), name=name, transform=T.NormalizeFeatures())[0]
        elif name == 'Actor':
            data = Actor(root='/tmp/actor', transform=T.NormalizeFeatures())[0]
        elif name == "Cornell" or name == "Texas" or name == "Wisconsin":
            data = WebKB(root='/tmp/' + str(name), name=name, transform=T.NormalizeFeatures())[0]
        elif name == 'squirrel' or name == 'chameleon':
            data = WikipediaNetwork(root='/tmp/' + str(name), name=name, transform=T.NormalizeFeatures())[0]

        indices = list(range(len(data.x)))

        train_indices = torch.tensor(indices[:int(0.7 * len(indices) + 1)])
        val_indices = torch.tensor(indices[int(0.7 * len(indices) + 1):int(0.8 * len(indices) + 1)])
        test_indices = torch.tensor(indices[int(0.8 * len(indices) + 1):])
        train_mask = torch.tensor([False] * len(indices))
        test_mask = torch.tensor([False] * len(indices))
        val_mask = torch.tensor([False] * len(indices))
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        val_mask[val_indices] = True
        return data, train_indices, val_indices, test_indices, train_mask, val_mask, test_mask


class Main:
    def __init__(self, name, conv, device, loss_function, mode):
        data, train_indices, test_indices, train_mask, test_mask = data_load(name)

        self.Conv = conv
        self.device = device
        self.x = data.x
        self.data = data.to(device)
        self.loss = loss_function
        self.mode = mode
        self.datasetname = name
        self.train_indices = train_indices  # torch.tensor(indices[:int(0.7*len(indices)+1)])
        self.test_indices = test_indices  # torch.tensor(indices[int(0.8*len(indices)+1):])
        self.train_mask = train_mask  # torch.tensor([False]*len(indices))
        self.test_mask = test_mask  # torch.tensor([False]*len(indices))
        self.flag = self.loss["flag_tosave"]
        print('END OF INIT MAIN')
        super(Main, self).__init__()

    def sampling(self, Sampler, epoch, nodes, loss, postfix):
        if (epoch == 0):
            if self.flag:
                if "alpha" in self.loss:
                    name_of_file = self.datasetname + "_samples_" + loss["Name"] + "_alpha_" + str(
                        loss["alpha"]) + '_' + str(postfix) + ".pickle"
                elif "betta" in self.loss:
                    name_of_file = self.datasetname + "_samples_" + loss["Name"] + "_betta_" + str(
                        loss["betta"]) + '_' + str(postfix) + ".pickle"
                else:

                    name_of_file = self.datasetname + "_samples_" + loss["Name"] + '_' + str(postfix) + ".pickle"

                if os.path.exists(f'{help_data}/' + str(name_of_file)):
                    with open(f'{help_data}/' + str(name_of_file), 'rb') as f:
                        self.samples = pickle.load(f)
                else:
                    self.samples = Sampler.sample(nodes, postfix)
                    with open(f'{help_data}/' + str(name_of_file), 'wb') as f:
                        pickle.dump(self.samples, f)
            else:
                self.samples = Sampler.sample(nodes, postfix)

    def train(self, model, data, optimizer, Sampler, train_loader, dropout, epoch, loss, postfix):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        # print('train loader',len(train_loader))

        if postfix == 'train':
            mask = self.train_mask
        elif postfix == 'full':
            mask = torch.BoolTensor([True] * len(data.x))
        else:
            mask = torch.BoolTensor([True] * len(data.x))

        if model.mode == 'unsupervised':
            if model.conv == 'GCN':
                arr = torch.nonzero(mask == True)
                indices_of_train_data = ([item for sublist in arr for item in sublist])
                # print('before',data.x)
                out = model.inference(data.to(self.device), dp=dropout)
                # print('after',out, sum(sum(out)))
                samples = self.sampling(Sampler, epoch, indices_of_train_data, loss, postfix)
                loss = model.loss(out[mask], self.samples)
                # print('loss',loss)
                total_loss += loss

            else:
                for batch_size, n_id, adjs in train_loader:

                    if len(train_loader.sizes) == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    out = model.forward(data.x[n_id.to(self.device)].to(self.device), adjs)
                    self.sampling(Sampler, epoch, n_id[:batch_size], loss, postfix)
                    loss = model.loss(out, self.samples)  # pos_batch.to(device), neg_batch.to(device))
                    total_loss += loss

            total_loss.backward()
            optimizer.step()
            return total_loss / len(train_loader), out

    @torch.no_grad()
    def test(self, model, data, test_loader, epoch, Sampler, loss):
        model.eval()
        total_loss = 0
        if model.mode == 'unsupervised':
            if model.conv == 'GCN':
                arr = torch.nonzero(self.test_mask == True)
                indices_of_train_data = ([item for sublist in arr for item in sublist])
                # print('before',data.x)
                out = model.inference(data.to(self.device), dp=0)
                # print('after',out, sum(sum(out)))
                self.sampling(Sampler, epoch, indices_of_train_data, loss, postfix='test')
                loss = model.loss(out[self.test_mask], self.samples)
                # print('loss',loss)
                total_loss += loss
            else:
                for batch_size, n_id, adjs in test_loader:
                    if len(test_loader.sizes) == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    out = model.forward(data.x[n_id.to(self.device)].to(self.device), adjs)

                    self.sampling(Sampler, epoch, n_id[:batch_size], loss, postfix='test')

                    loss = model.loss(out, self.samples)  # pos_batch.to(device), neg_batch.to(device))
                    total_loss += loss

            return total_loss / len(test_loader)

    def run(self, params):
        hidden_layer = params['hidden_layer']
        out_layer = params['out_layer']
        dropout = params['dropout']
        size = params['size of network, number of convs']
        learning_rate = params['lr']
        print('i am in run')
        # hidden_layer_for_classifier=params['hidden_layer_for_classifier']
        # alpha_for_classifier = params['alpha_for_classifier']
        # learning_rate_for_classifier = params['learning_rate_for_classifier']
        # n_layers_for_classifier = params['n_layers_for_classifier']

        # hidden_layer=64,out_layer=128,dropout=0.0,size=1,learning_rate=0.001,c=100

        self.data.edge_index = self.data.edge_index.type(torch.LongTensor)
        ns = NegativeSampler(data=self.data)  # это для того чтоб тестовые негативные примеры не включали

        all_edges = self.data.edge_index.T.tolist()
        train_edges = []
        test_edges = []
        indices_train_edges = random.sample(range(len(all_edges)), int(len(all_edges) * 0.8))
        for i, edge in enumerate(all_edges):
            if i in indices_train_edges:
                train_edges.append(edge)
            else:
                test_edges.append(edge)

        self.data.edge_index = torch.LongTensor(train_edges).T

        train_loader = NeighborSampler(self.data.edge_index, node_idx=torch.BoolTensor([True] * len(self.data.x)),
                                       batch_size=int(len(self.data.x)), sizes=[-1] * size)

        Sampler = self.loss["Sampler"]
        LossSampler = Sampler(self.datasetname, self.data, device=self.device,
                              mask=torch.BoolTensor([True] * len(self.data.x)), loss_info=self.loss, help_dir=help_data)
        model = Net(dataset=self.data, mode=self.mode, conv=self.Conv, loss_function=self.loss, device=self.device,
                    hidden_layer=hidden_layer, out_layer=out_layer, num_layers=(size), dropout=dropout)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        for epoch in range(100):
            loss, out = self.train(model, self.data, optimizer, LossSampler, train_loader, dropout, epoch, self.loss,
                                   postfix='trainLP')

        np.save('../data_help/' + str(self.datasetname) + '_' + str(self.loss['Name']) + '_emb.npy',
                out.detach().cpu().numpy())

        positive_edges = test_edges
        ###вот отюсда доделывать

        num_neg_samples_test = int(len(positive_edges) / len(self.data.x))
        print('first num neg samples test', (num_neg_samples_test))
        num_neg_samples_test = num_neg_samples_test if num_neg_samples_test > 0 else 1

        neg_samples_test = ns.negative_sampling(torch.LongTensor(list(range(len(self.data.x)))),
                                                num_negative_samples=num_neg_samples_test)
        print('num_negative_samples_test', num_neg_samples_test)
        print('len', len(positive_edges), len(neg_samples_test))

        if (num_neg_samples_test == 1) and (len(neg_samples_test) > len(positive_edges)):
            ind = torch.randperm(len(positive_edges))
            neg_samples_test = neg_samples_test[ind]
        emb_norm = torch.nn.functional.normalize(torch.tensor(out.detach().cpu()))

        pred_test = []
        for edge in positive_edges:
            pred_test.append((torch.dot(emb_norm[edge[0]], emb_norm[edge[1]])))
        # print(torch.sigmoid(torch.dot(emb_norm[edge[0]],emb_norm[edge[1]])))

        for edge in neg_samples_test:
            pred_test.append((torch.dot(emb_norm[edge[0]], emb_norm[edge[1]])))

        true_test = [1] * len(positive_edges) + [0] * len(neg_samples_test)

        return roc_auc_score(true_test, pred_test)


class MainOptuna(Main):
    def objective(self, trial):
        # Integer parameter
        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128, 256])
        out_layer = trial.suggest_categorical("out_layer", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        size = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        Conv = self.Conv
        learning_rate = trial.suggest_float("lr", 5e-3, 1e-2)
        #   learning_rate_for_classifier =trial.suggest_float("learning_rate_for_classifier",5e-3,1e-2)
        #  n_layers_for_classifier = trial.suggest_categorical("n_layers_for_classifier", [1,2,3])
        # alpha_for_classifier = trial.suggest_categorical("alpha_for_classifier",  [0.001, 0.01, 0.1,0.3,0.5,0.7,0.9,1,10,20,30,100])
        # c =trial.suggest_categorical("c",  [0.001, 0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,20,30,100])
        # hidden_layer_for_classifier = trial.suggest_categorical("hidden_layer_for_classifier", [32,64,128,256])
        # варьируем параметры
        loss_to_train = {}
        for name in self.loss:

            if type(self.loss[name]) == list:
                if len(self.loss[name]) == 3:
                    var = trial.suggest_int(name, self.loss[name][0], self.loss[name][1], step=self.loss[name][2])
                    loss_to_train[name] = var
                elif len(self.loss[name]) == 2:
                    var_2 = trial.suggest_float(name, self.loss[name][0], self.loss[name][1])
                    loss_to_train[name] = var_2
                else:
                    var_3 = trial.suggest_categorical(name, self.loss[name])
                    loss_to_train[name] = var_3
            else:
                loss_to_train[name] = self.loss[name]
        if name == 'q' and type(self.loss[name]) == list:
            var_5 = trial.suggest_categorical('p', self.loss['p'])
            var_4 = trial.suggest_categorical('q', self.loss[name])
            if var_4 > 1:
                var_4 = 1
            if var_5 < var_4:
                var_5 = var_4
            loss_to_train['q'] = var_4
            loss_to_train['p'] = var_5

        Sampler = loss_to_train["Sampler"]
        model = Net(dataset=self.data, mode=self.mode, conv=Conv, loss_function=loss_to_train, device=self.device,
                    hidden_layer=hidden_layer, out_layer=out_layer, num_layers=size, dropout=dropout)
        self.data.edge_index = self.data.edge_index.type(torch.LongTensor)

        train_loader = NeighborSampler(self.data.edge_index, batch_size=int(sum(self.train_mask)),
                                       node_idx=self.train_mask, sizes=[-1] * size)
        test_loader = NeighborSampler(self.data.edge_index, batch_size=int(sum(self.test_mask)),
                                      node_idx=self.test_mask, sizes=[-1] * size)
        LossSampler = Sampler(self.datasetname, self.data, device=self.device, mask=self.train_mask,
                              loss_info=loss_to_train, help_dir=help_data)
        LossSamplerTest = Sampler(self.datasetname, self.data, device=self.device, mask=self.test_mask,
                                  loss_info=loss_to_train, help_dir=help_data)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(50):
            loss, _ = self.train(model, self.data, optimizer, LossSampler, train_loader, dropout, epoch, loss_to_train,
                                 postfix='train')
        loss_test = self.test(model=model, data=self.data, epoch=0, test_loader=test_loader, Sampler=LossSamplerTest,
                              loss=loss_to_train)
        trial.report(loss_test, epoch)
        return loss_test

    def run(self, number_of_trials):
        study = optuna.create_study(direction="minimize",
                                    study_name=self.loss["Name"] + " loss," + str(self.Conv) + " conv")
        study.optimize(self.objective, n_trials=number_of_trials)
        #  joblib.dump(study, "study_" + str(self.datasetname) + str(self.Conv) +'_'+ str(self.loss["Name"])+ ".pkl")

        return study

