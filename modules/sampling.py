import abc
import collections
import math
import os
import pickle
from datetime import datetime
import random

import numpy as np
import torch
from modules.negativeSampling import NegativeSampler
from torch_sparse import SparseTensor


import torch_cluster  # noqa
RW = torch.ops.torch_cluster.random_walk


from torch_geometric.utils import subgraph


class Sampler:
    def __init__(self, datasetname, data, device, mask, loss_info, **kwargs):
        self.device = device
        self.datasetname = datasetname
        self.data = data.to(self.device)

        self.mask = mask
        self.NS = NegativeSampler(self.data, self.device)
        self.loss = loss_info
        super(Sampler, self).__init__()

    def edge_index_to_adj_train(self, batch):
        x_new = torch.sort(batch).values
        x_new = x_new.tolist()
        mapping = {}
        i=0
        for value in (x_new):
            if value not in mapping:
                mapping[value] = i
                i+=1

        A = torch.zeros((len(x_new), len(x_new)), dtype=torch.long)  # .to(self.device)

        edge_index_0 = self.data.edge_index[0].tolist()
        edge_index_1 = self.data.edge_index[1].tolist()
        for j, i in enumerate(edge_index_0):
            if i in x_new:
                if edge_index_1[j] in x_new:
                    A[mapping[i]][mapping[edge_index_1[j]]] = 1

        return A,mapping

    def edge_index_to_adj_train_old(self, mask, batch):
        x_new = torch.tensor(np.where(mask == True)[0], dtype=torch.int32)
        A = torch.zeros((len(x_new), len(x_new)), dtype=torch.long)

        edge_index_0 = self.data.edge_index[0].to("cpu")
        edge_index_1 = self.data.edge_index[1].to("cpu")

        for j, i in enumerate(edge_index_0):
            if i in x_new:
                if edge_index_1[j] in x_new:
                    A[i][edge_index_1[j]] = 1
        return A

    @abc.abstractmethod
    def sample(self, batch, postfix,**kwargs):
        pass

class SamplerWithNegSamples(Sampler):
    def __init__(self, datasetname, data, device, mask, loss_info, **kwargs):
        Sampler.__init__(self, datasetname, data, device, mask, loss_info, **kwargs)
        self.num_negative_samples = self.loss["num_negative_samples"]

    def sample(self, batch,postfix):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return (self.pos_sample(batch,postfix), self.neg_sample(batch))

    @abc.abstractmethod
    def pos_sample(self, batch,postfix):
        pass

    def neg_sample(self, batch):
        len_batch = len(batch)
        a, _ = subgraph(batch.tolist(), self.data.edge_index)
        neg_batch = self.NS.negative_sampling(batch, num_negative_samples=self.num_negative_samples)
        return neg_batch  # %len_batch


class SamplerRandomWalk(SamplerWithNegSamples):
    def __init__(self, datasetname, data, device, mask, loss_info, **kwargs):
        SamplerWithNegSamples.__init__(self, datasetname, data, device, mask, loss_info, **kwargs)

        self.loss = loss_info
        self.p = self.loss["p"]
        self.q = self.loss["q"]
        self.walk_length = self.loss["walk_length"]
        self.walks_per_node = self.loss["walks_per_node"]
        self.context_size = (
            self.loss["context_size"] if self.walk_length >= self.loss["context_size"] else self.walk_length
        )

    def neg_sample(self, batch):

        len_batch = len(batch)
        a, _ = subgraph(batch.tolist(), self.data.edge_index)
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        # print(c, batch,self.num_negative_samples)
        neg_batch = self.NS.negative_sampling(batch, num_negative_samples=self.num_negative_samples)
        return neg_batch

    def pos_sample(self, batch,postfix):
        name_of_samples = (
                self.datasetname
                + "_"
                + str(self.walk_length)
                + "_"
                + str(self.walks_per_node)
                + "_"
                + str(self.context_size)
                + "_"
                + str(self.p)
                + "_"
                + str(self.q)
                + '_'
                + str(postfix)
                + ".pickle"
        )
        if os.path.exists(name_of_samples):
            with open(name_of_samples, "rb") as f:
                pos_samples = pickle.load(f)
        else:
            d = datetime.now()
            len_batch = len(batch)

            x_new = batch.tolist()
            mapping = {}
            i = 0
            for value in (x_new):
                if value not in mapping:
                    mapping[value] = i
                    i += 1
            a, _ = subgraph(batch.to(self.device), self.data.edge_index)
            row, col = a.to('cpu')
           # rowli = row.tolist()
            #colli = col.tolist()

          #  row_new = []
         #   col_new = []

            #for elem in rowli:
            #    row_new.append(mapping[elem])
            #for elem2 in colli:
            #    col_new.append(mapping[elem2])

#            row = torch.LongTensor(row_new).to('cuda')
 #           col = torch.LongTensor(col_new).to('cuda')
            row = row.apply_(lambda x: mapping[x])  # .type(torch.IntTensor)
            col = col.apply_(lambda x: mapping[x])  # .type(torch.IntTensor)

            d1 = datetime.now()
            adj = SparseTensor(row=row, col=col, sparse_sizes=(len_batch, len_batch)).to(self.device)

            rowptr, col, _ = adj.csr()
            d2 = datetime.now()
            start = batch.repeat(self.walks_per_node).to(self.device)
            print('Random Walk')
            rw = RW(rowptr, col, start, self.walk_length, self.p, self.q)
            print('Random Walk2')
            if not isinstance(rw, torch.Tensor):
                rw = rw[0]
            walks = []
            num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
            for j in range(num_walks_per_rw):
                walks.append(
                    rw[:, j: j + self.context_size]
                )  # теперь у нас внутри walks лежат 12 матриц размерам 10*1
            pos_samples = torch.cat(walks, dim=0)  # .to(self.device)
        # with open(name_of_samples,'wb') as f:
        #    pickle.dump(pos_samples,f)
        # if os.stat(name_of_samples).st_size/1000000 >100:
        #   with open('Chameleon_20_5_20_1_1.pickle','rb') as f:
        #      pos_samples = pickle.load(f)
        # print('nope')
        return pos_samples


class SamplerContextMatrix(SamplerWithNegSamples):
    def __init__(self, datasetname, data, device, mask, loss_info, help_dir, **kwargs):
        SamplerWithNegSamples.__init__(self, datasetname, data, device, mask, loss_info, **kwargs)
        self.loss = loss_info
        self.help_dir = help_dir
        if self.loss["C"] == "PPR":
            self.alpha = round(self.loss["alpha"], 1)

    def pos_sample(self, batch, postfix, **kwargs):

        d_pb = datetime.now()
        batch = batch
        pos_batch = []
        d = datetime.now()
        if self.loss["C"] == "Adj" and (self.loss["Name"] == "LINE" or self.loss["Name"] == "Force2Vec"):
            name = f"{self.help_dir}/pos_samples_"+str(self.loss["Name"])+'_'+self.datasetname+'_'+str(postfix)+".pickle"
            if os.path.exists(name):
                with open(name, "rb") as f:
                    pos_batch = pickle.load(f)
            else:
                A,mapping = self.edge_index_to_adj_train(batch)
                pos_batch = self.convert_to_samples(batch,mapping, A)
                with open(name, "wb") as f:
                    pickle.dump(pos_batch, f)
        elif self.loss["C"] == "Adj" and self.loss["Name"] == "VERSE_Adj":
            name = f"{self.help_dir}/pos_samples_VERSEAdj_" + self.datasetname+ '_' +str(postfix) + ".pickle"
            if os.path.exists(name):
                with open(name, "rb") as f:
                    pos_batch = pickle.load(f)
            else:
                Adj,mapping = self.edge_index_to_adj_train(batch)
                Adj=Adj.type(torch.FloatTensor)
                A = (Adj / sum(Adj))
                #A[torch.isinf(A)] = 0
                #A[torch.isnan(A)] = 0
                pos_batch = self.convert_to_samples(batch, mapping, A)
                with open(name, "wb") as f:
                    pickle.dump(pos_batch, f)
        elif self.loss["C"] == "Adj" and self.loss["Name"] == "GraRep":
            name = f"{self.help_dir}/pos_samples_GraRep_" + self.datasetname+ '_' +str(postfix) + ".pickle"
            if os.path.exists(name):
                with open(name, "rb") as f:
                    pos_batch = pickle.load(f)
            else:
                Adj,mapping = self.edge_index_to_adj_train(batch)
                Adj=Adj.type(torch.FloatTensor)
                A = Adj
                for _ in range(5):
                    A += A * Adj

                A[torch.isinf(A)] = 0
                A[torch.isnan(A)] = 0
                pos_batch = self.convert_to_samples(batch, mapping, A)
                with open(name, "wb") as f:
                    pickle.dump(pos_batch, f)

        elif self.loss["C"] == "SR":
            SimRankName = f"{self.help_dir}/SimRank" + self.datasetname+ '_' +str(postfix) + ".pickle"
            if os.path.exists(SimRankName):
                with open(SimRankName, "rb") as f:
                    A = pickle.load(f)
            else:
                Adj, _ = subgraph(batch, self.data.edge_index)
                row, col = Adj
                row = row.to(self.device)
                col = col.to(self.device)
                ASparse = SparseTensor(row=row, col=col, sparse_sizes=(len(batch), len(batch)))
                r = 200
                length = list(map(lambda x: x * int(r / 100), [22, 17, 14, 10, 8, 6, 5, 4, 3, 11]))  # O(t)
                mask = []
                for i, l in enumerate(length):  # max(l) = (1-sqrt(c)); O((1-sqrt(c))*t^2)
                    mask1 = torch.zeros([l, 10])  # O( (1-sqrt(c)) *t)
                    mask1.t()[: (i + 1)] = 1  # O( 3*(1-sqrt(c)) *t)=O((1-sqrt(c)) *t)
                    mask.append(mask1)
                mask = torch.cat(mask)  # O((1-sqrt(c)) *t^2)
                mask_new = 1 - mask
                A = self.find_sim_rank_for_batch_torch(batch, ASparse, self.device, mask, mask_new, r)

                with open(SimRankName, "wb") as f:
                    pickle.dump(A, f)
            samples_name = f"{self.help_dir}/samples_simrank_" + self.datasetname + '_' +str(postfix)+ ".pickle"
            if os.path.exists(samples_name):

                with open(samples_name, "rb") as f:
                    pos_batch = pickle.load(f)

            else:
                pos_batch = []
                batch_l = batch.tolist()
                for x in batch_l:
                    # print('{}/{}'.format(x,len(batch_l)))
                    for j in batch_l:
                        # print(x,j,'in',len(batch_l))
                        if A[int(x)][int(j)] != torch.tensor(0):
                            pos_batch.append([int(x), int(j), A[x][j]])

                pos_batch = torch.tensor(pos_batch)
                with open(samples_name, "wb") as f:
                    pickle.dump(pos_batch, f)

        elif self.loss["C"] == "PPR":
            alpha = self.alpha
            name_of_file = f"{self.help_dir}/pos_samples_VERSEPPR_" + str(alpha) + "_" + self.datasetname + '_' + str(postfix) + ".pickle"
            if os.path.exists(name_of_file):
                with open(name_of_file, "rb") as f:
                    pos_batch = pickle.load(f)
            else:
                Adg, mapping = self.edge_index_to_adj_train(batch)
                Adg = Adg.type(torch.FloatTensor)
                #invD[torch.isinf(invD)] = 0
                A = (1 - alpha) * torch.inverse(torch.diag(torch.ones(len(Adg))) - alpha * Adg/Adg.sum(dim=0))
                pos_batch = self.convert_to_samples(batch, mapping, A)
                with open(name_of_file, "wb") as f:
                    pickle.dump(pos_batch, f)

        return pos_batch

    @staticmethod
    def convert_to_samples(batch, mapping, A):
        pos_batch = []
        batch_l = batch.tolist()
        for x in batch_l:
            # print('{}/{}'.format(x,len(batch_l)))
            for j in batch_l:
                # print(x,j,'in',len(batch_l))
                if A[mapping[x]][mapping[j]] != torch.tensor(0):
                    pos_batch.append([int(mapping[x]), int(mapping[j]), A[mapping[x]][mapping[j]]])

        return torch.tensor(pos_batch)

    # A=A.to(self.device)
    # t = len(A)
    # pos_batch = torch.Tensor( torch.nonzero(A).size(0), 3 ).to(self.device)
    # p = 0
    # for f,x in enumerate(batch):
    #    for j in range(t):
    #        if A[f][j] != torch.tensor(0):
    #            pos_batch[p][0] = (f)
    #            pos_batch[p][1] =(j)
    #            pos_batch[p][2] = (A[f][j])
    #            p+=1
    # return pos_batch

    def find_sim_rank_for_batch_torch(self, batch, Adj, device, mask, mask_new, r):
        t = 10
        c = torch.sqrt(torch.tensor(0.6))
        # approx with SARW
        batch = batch.to(device)
        Adj = Adj.to(device)
        SimRank = torch.zeros(len(batch), len(batch)).to(device)  # O(n^2)
        for u in batch:  # O(n^3*(tr))
            print("{}/{}".format(u, len(batch)))
            d = datetime.now()
            for nei in batch:
                prob_i = torch.zeros(t).to(device)  # O(t)

                d_rw = datetime.now()
                pi_u = Adj.random_walk(u.repeat(r).flatten(), walk_length=t)  # O(tnr)
                pi_v = Adj.random_walk(nei.repeat(r).flatten(), walk_length=t)  # O(tnr)
                pi_u = pi_u[:, 1:]
                pi_v = pi_v[:, 1:]
                mask = mask.to(self.device)
                mask_new = mask_new.to(self.device)
                pi_u = pi_u * mask - mask_new  # O(r*t)
                pi_v = pi_v * mask - mask_new  # O(r*t)

                #  d_sr = datetime.now()
                a1 = pi_u == pi_v  # O(r*t)
                a2 = pi_u != -1  # O(r*t)
                a3 = pi_v != -1  # O(r*t)
                a_to_compare = a1 * a2 * a3  # O(r*t)
                SR = len(torch.unique(a_to_compare.nonzero(as_tuple=True)[0]))  # O()
                SimRank[u][nei] = SR / r  #

        return SimRank


class SamplerFactorization(Sampler):
    def sample(self, batch, postfix, **kwargs):

        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)

        A, mapping = self.edge_index_to_adj_train(batch)
        if self.loss["loss var"] == "Factorization":
            if self.loss["C"] == "Adj":
                C = A
            elif self.loss["C"] == "CN":
                A = A.type(torch.FloatTensor).to(self.device)
                C = torch.matmul(A, A)
            elif self.loss["C"] == "AA":
                D = torch.diag(1 / (sum(A) + sum(A.t())))
                A = A.type(torch.FloatTensor)
              #  D[torch.isinf(D)] = 0
               # D[torch.isnan(D)] = 0
                C = torch.matmul(torch.matmul(A, D), A)

            elif self.loss["C"] == "Katz":

                A = A.type(torch.FloatTensor)
               # A[torch.isinf(A)] = 0
               # A[torch.isnan(A)] = 0
                betta = self.loss["betta"]
                I = torch.diag(torch.ones(len(A)))
                # print('IAB',I,A,betta)
                # print('I-BA',I-betta*A)
                inv = torch.linalg.inv(I - betta * A)
                vals=torch.linalg.eigvals(A)
                # print(sum(sum(inv)),sum(inv),inv)
                C = betta * torch.matmul(inv, A)
            elif self.loss["C"] == "RPR":
                alpha = self.loss["alpha"]
                A = A.type(torch.FloatTensor)
                N = A.sum(dim=0)
                # Создаем матрицу P, где каждый элемент A_ij, равный 1, заменяется на 1/N(j)
                P = A / N
                #invD[torch.isinf(invD)] = 0
                # print(torch.inverse(torch.diag(torch.ones(len(A))))
                C = (1 - alpha) * torch.inverse(torch.diag(torch.ones(len(A))) - alpha * P)

            return C
        else:
            return A


class SamplerAPP(SamplerWithNegSamples):
    def __init__(self, datasetname, data, device, mask, loss_info, **kwargs):
        SamplerWithNegSamples.__init__(self, datasetname, data, device, mask, loss_info, **kwargs)
        self.device = device  # if torch.cuda.is_available() else 'cpu')

        self.alpha = self.loss["alpha"]
        self.r = 200
        self.num_negative_samples *= 10

        new_edge_index = []
        li = data.edge_index.t().tolist()
        for edge in li:
            if edge not in new_edge_index:
                new_edge_index.append(edge)
            if [edge[1], edge[0]] not in new_edge_index:
                new_edge_index.append([edge[1], edge[0]])
        new_edge_index = torch.tensor(new_edge_index).t()
        self.data.edge_index = new_edge_index

    def sample(self, batch, postfix, **kwargs):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return self.pos_sample(batch, postfix), self.neg_sample(batch)

    def pos_sample(self, batch, postfix,  **kwargs):

        name_pos_samples = "../data_help/APP_" + self.datasetname + "_" + str(self.alpha) + '_' + str(postfix) + "_.pickle"
        if os.path.exists(name_pos_samples):
            with open(name_pos_samples, "rb") as f:
                pos_batch = pickle.load(f)
        else:
            d_pb = datetime.now()
            batch = batch.to(self.device)
            len_batch = len(batch)
            mask = torch.tensor([False] * len(self.data.x))
            mask[batch.tolist()] = True
            d = datetime.now()

            x_new = sorted(batch.tolist())
            mapping = {}
            i = 0
            for value in (x_new):
                if value not in mapping:
                    mapping[value] = i
                    i += 1

            a, _ = subgraph(batch, self.data.edge_index)
            # print(a)
            # g = nx.Graph()
            # g.add_edges_from((a.t()).tolist())
            # for com in nx.connected_components(g):
            #    print(len(com))

            row, col = a

            row = row.tolist()
            col = col.tolist()

            row_new = []
            col_new = []

            for elem in row:
                row_new.append(mapping[elem])
            for elem in col:
                col_new.append(mapping[elem])

            #           col = torch.LongTensor(col_new).to('cuda')
          #  row = row.apply_(lambda x: mapping[x])  # .type(torch.IntTensor)
          #  col = col.apply_(lambda x: mapping[x])  # .type(torch.IntTensor)
            row = torch.LongTensor(row_new)
            col = torch.LongTensor(col_new)
            row = row.to(self.device)
            col = col.to(self.device)

            # start  = torch.tensor(list(set(row.tolist()) & set(col.tolist()) & set(batch.tolist())),dtype=torch.long)
            ASparse = SparseTensor(row=row, col=col, sparse_sizes=(len_batch, len_batch))
            pos_dict = self.find_PPR_approx(batch, ASparse, self.device, self.r, self.alpha, row)
            pos_batch = []
            for pos_pair in pos_dict:
                pos_row = list(pos_pair)
                pos_row.append(pos_dict[pos_pair])
                pos_batch.append(pos_row)
            with open(name_pos_samples, "wb") as f:
                pickle.dump(pos_batch, f)
        return torch.tensor(pos_batch)

    def find_PPR_approx(self, batch, Adj, device, r, alpha, row):
        N = math.ceil(math.log(1 / (r * alpha), (1 - alpha)))
        length = list(map(lambda x: int((1 - alpha) ** x * alpha * r), list(range(N - 1))))
        r = sum(length)
        dict_data = dict()
        # device = 'cpu'
        for u in batch:
            if len(torch.where(row == u)[0]) > 0:
                d = datetime.now()
                # print(u.to(device).repeat(r).flatten())
                pi_u = Adj.random_walk(u.to(device).repeat(r).flatten(), walk_length=N)
                split = torch.split(pi_u, length)
                pos_batch = []

                for i, seg in enumerate(split):
                    pos_samples = collections.Counter(seg[:, (i + 1)].tolist())
                    for pos_sample in pos_samples:
                        if (int(seg[0][0]), int(pos_sample)) in dict_data:
                            dict_data[(int(seg[0][0]), int(pos_sample))] += pos_samples[pos_sample]
                        elif (int(pos_sample), int(seg[0][0])) in dict_data:
                            dict_data[(int(pos_sample), int(seg[0][0]))] += pos_samples[pos_sample]
                        else:
                            dict_data[(int(seg[0][0]), int(pos_sample))] = 1
        return dict_data

class SamplerRNWE(Sampler):
    def __init__(self, datasetname, data, device, mask, loss_info, **kwargs):
        Sampler.__init__(self, datasetname, data, device, mask, loss_info, **kwargs)

        self.loss = loss_info
        self.walk_length = self.loss["walk_length"]
        self.alpha = self.loss["alpha"]
        self.walks_per_node = self.loss["walks_per_node"]

    def _restart_random_walk(self, start_nodes, adj, walk_length = 0, alpha=0.1):
        num_nodes = adj.shape[0]
        normalized_adj_matrix = self.normalize_adjacency_matrix(adj)
        walks = torch.empty((len(start_nodes), walk_length), dtype=torch.long)
        current_positions = start_nodes.clone().detach() #torch.tensor(start_nodes, dtype=torch.long)
        walks[:, 0] = current_positions

        for step in range(1, walk_length):
            # Определение, кто из узлов будет перемещаться дальше
            moves = torch.rand(len(start_nodes)) > alpha
            for i, move in enumerate(moves):
                if move:
                    # Выбор следующей вершины с учетом вероятностей перехода
                    current_node = current_positions[i]
                    probabilities = normalized_adj_matrix[current_node]
                    next_node = torch.multinomial(probabilities, 1).item()
                    current_positions[i] = next_node
                else:
                    # Рестарт к начальной вершине
                    current_positions[i] = start_nodes[i]

            walks[:, step] = current_positions

        return walks # input(center_node), targets(context_nodes), neg_targets(neg_nodes)
    def normalize_adjacency_matrix(self, adj_matrix):
        # Преобразование типов для поддержки torch.multinomial
        adj_matrix_float = adj_matrix.float()
        adj_matrix_float += 1e-9 # для изолированных вершин чтоб не было инф
        # Нормализация строк матрицы смежности с добавлением небольшой константы к сумме для избежания деления на 0
        row_sums = adj_matrix_float.sum(dim=1, keepdim=True) + 1e-6  # Добавляем небольшую константу к суммам строк
        normalized_adj_matrix = adj_matrix_float / row_sums
        return normalized_adj_matrix
    def neg_sample(self, walks,postfix): #это оч долго
        name_of_samples = (
                self.datasetname
                + "_"
                + str(self.walk_length)
                + "_"
                + str(self.walks_per_node)
                + '_'
                + str(self.alpha)
                + '_'
                + str(postfix)
                + str('_negative')
                + ".pickle"
        )
        if os.path.exists(name_of_samples):
            with open(name_of_samples, "rb") as f:
                negative_samples = pickle.load(f)
        else:
            unique_nodes_per_walk = walks.unique(dim=1)
            # Инициализация матрицы негативных примеров
            negative_samples = torch.empty_like(walks)
            num_nodes = len(set(walks[:,0].tolist()))
            for i in range(len(walks)):
                # Доступные вершины для негативных примеров
                available_nodes = torch.tensor([node for node in range(num_nodes) if node not in unique_nodes_per_walk[i]],
                                               dtype=torch.long)

                # Генерация негативных примеров
                for j in range(1, walks.shape[1]):  # Пропускаем первый столбец, так как он для стартовых вершин
                    negative_samples[i, j] = available_nodes[torch.randint(0, len(available_nodes), (1,))]

            # Установка стартовых вершин на первые позиции
            negative_samples[:, 0] = walks[:,0]
            with open(name_of_samples,'wb') as f:
                pickle.dump(negative_samples,f)
        return negative_samples

    def pos_sample(self, batch, postfix):
        name_of_samples = (
                self.datasetname
                + "_"
                + str(self.walk_length)
                + "_"
                + str(self.walks_per_node)
                + '_'
                +  str(self.alpha)
                + '_'
                + str(postfix)
                + ".pickle"
        )
        if os.path.exists(name_of_samples):
            with open(name_of_samples, "rb") as f:
                pos_samples = pickle.load(f)
        else:
            len_batch = len(batch)
            x_new = batch.tolist()
            mapping = {}
            i = 0
            for value in (x_new):
                if value not in mapping:
                    mapping[value] = i
                    i += 1

            start = batch.repeat(self.walks_per_node)
            new_start = torch.empty_like(start)
            for key, value in mapping.items():
                new_start[start == key] = value

            adj, _ = self.edge_index_to_adj_train(batch)
            pos_samples = self._restart_random_walk(new_start, adj, self.walk_length, self.alpha)
            #pos_samples = torch.cat(walks)  # .to(self.device)
           # print('pos_samples',pos_samples)
            with open(name_of_samples,'wb') as f:
                pickle.dump(pos_samples,f)

        return pos_samples

    def sample(self, batch,postfix):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        pos_samples = self.pos_sample(batch,postfix)
        neg_samples = self.neg_sample(pos_samples,postfix)
        return (pos_samples, neg_samples)
