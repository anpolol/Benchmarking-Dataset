import os
import pandas as pd
import torch

from MainTrain import Main, MainOptuna
from modules.sampling import SamplerContextMatrix, SamplerRandomWalk, SamplerFactorization, SamplerAPP, SamplerRNWE

synthetic = True
if synthetic:
    datasets_names=[]
    for l_a_trgt in [0.5]:
                for f_a_trgt in  [x / 10 for x in range(10)]:
                    for cl_trgt in [x / 20 for x in range(20)]:
                        for asp_trgt in [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5]:
                            for a_deg_trgt in  [x for x in range(2, 10)]:
                                datasets_names.append((l_a_trgt,f_a_trgt,cl_trgt,asp_trgt,a_deg_trgt))

#loss functions
LapEigen = {"Name": "LaplacianEigenMaps", "C":"Adj","loss var": "Laplacian EigenMaps","flag_tosave":True,"Sampler" :SamplerFactorization,"lmbda": [0.0,1.0]}
GraphFactorization = {"Name": "Graph Factorization","C":"Adj","loss var": "Factorization","flag_tosave":False,"Sampler" :SamplerFactorization,"lmbda": [0.0,1.0]}
HOPE_CN = {"Name": "HOPE_CommonNeighbors", "C":"CN","loss var": "Factorization","flag_tosave":False,"Sampler" :SamplerFactorization,"lmbda": [0.0,1.0]}
HOPE_AA = {"Name": "HOPE_AdamicAdar","C":"AA","loss var": "Factorization","flag_tosave":True,"Sampler" :SamplerFactorization,"lmbda": [0.0,1.0]}

#--------------------------------------------------------------------
VERSE_PPR =  {"Name": "VERSE_PPR","C": "PPR","num_negative_samples":[1, 6, 11, 16, 21],"loss var": "Context Matrix","flag_tosave":False,"alpha": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"Sampler" :SamplerContextMatrix,"lmbda": [0.0,1.0]}
VERSE_Adj =  {"Name": "VERSE_Adj","C": "Adj","num_negative_samples":[1, 6, 11, 16, 21],"loss var": "Context Matrix","flag_tosave":False,"Sampler" :SamplerContextMatrix,"lmbda": [0.0,1.0]}
VERSE_SR =  {"Name": "VERSE_SimRank","C": "SR","num_negative_samples":[1, 6, 11, 16, 21],"loss var": "Context Matrix","flag_tosave":False,"Sampler":SamplerContextMatrix,"lmbda": [0.0,1.0]}
DeepWalk = {"Name": "DeepWalk","walk_length":[5, 10, 15, 20],"walks_per_node":[5, 10, 15, 20],"num_negative_samples":[1,6, 11, 16, 21],"context_size" : [5, 10, 15, 20],"p":1,"q":1,"loss var": "Random Walks","flag_tosave":False,"Sampler" : SamplerRandomWalk } #Проблемы с памятью после того, как увеличила количество тренировочных данных
Node2Vec = {"Name": "Node2Vec","walk_length":[5, 10, 15, 20],"walks_per_node":[5, 10, 15, 20],"num_negative_samples":[1,6, 11, 16, 21],"context_size" : [5, 10, 15, 20],"p": [0.25, 0.50, 1, 2, 4] ,"q":[0.25, 0.50, 1, 2, 4], "loss var": "Random Walks","flag_tosave":False,"Sampler": SamplerRandomWalk}#то же самое
APP ={"Name": "APP","C": "PPR","num_negative_samples":[1, 6, 11, 16, 21],"loss var": "Context Matrix","flag_tosave":True,"alpha": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"Sampler" :SamplerAPP}
HOPE_Katz = {"Name": "HOPE_Katz","C":"Katz","loss var": "Factorization","flag_tosave":True,"betta": [0.1,0.2,0.3,0.4,0.5],"Sampler" :SamplerFactorization,"lmbda": [0.0,1.0]} #проверить

HOPE_RPR = {"Name": "HOPE_RPR","C":"RPR","loss var": "Factorization","flag_tosave":True,"alpha": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"Sampler" :SamplerFactorization,"lmbda": [0.0,1.0]} #проверить

LINE = {"Name": "LINE","C": "Adj","num_negative_samples":[1, 6, 11, 16, 21],"loss var": "Context Matrix","flag_tosave":False,"Sampler" :SamplerContextMatrix,"lmbda": [0.0,1.0]}

GraRep = {"Name": "GraRep","C": "Adj","num_negative_samples":[1, 6, 11, 16, 21],"loss var": "GraRep","flag_tosave":False,"Sampler" :SamplerContextMatrix,"lmbda": [0.0,1.0]}

Force2Vec = {"Name": "Force2Vec","C": "Adj","num_negative_samples":[1, 6, 11, 16, 21],"loss var": "Force2Vec","flag_tosave":False,"Sampler" :SamplerContextMatrix,"lmbda": [0.0,1.0]}

RNWE = {"Name": "RNWE","walk_length":[5, 10, 15, 20],"walks_per_node":[5, 10, 15, 20], "alpha": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"lmbda": [1,5,10],"gamma": [1,5,10], "loss var": "RNWE","flag_tosave":False,"Sampler": SamplerRNWE}#то же самое

if os.path.exists('../results/transductive_results.csv'):
    dataframe = pd.read_csv('../results/transductive_results.csv')
    dataframe = dataframe.drop(columns=['Unnamed: 0'])
else:
    dataframe = pd.DataFrame(columns=['loss', 'conv', 'f','cl','asp','ad','dataset' , 'coherence', 'stable_rank','cond_number', 'self_cluster', 'best_values'])
    dataframe.to_csv('../results/transductive_results.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for loss in [VERSE_Adj, VERSE_PPR, VERSE_SR, DeepWalk, Node2Vec, APP, HOPE_AA, HOPE_CN, HOPE_Katz, HOPE_RPR, LINE, LapEigen, GraphFactorization, GraRep, Force2Vec, RNWE]:
    loss_name = loss["Name"]
    for (l, f, cl, asp, ad) in datasets_names:
        for conv in ['transductive']:
            name = "".join(list(map(lambda x: str(x), [l, f, cl, asp, ad])))
            if os.path.exists('../dataset/graph_' + str(name) + '_attr.npy'):
                if len(dataframe[(dataframe['loss'] == loss_name) & (dataframe['conv'] == conv) & (
                        dataframe['dataset'] == name)]) == 0:
                    MO = MainOptuna(name=name, conv=conv, device=device, loss_function=loss, mode='unsupervised')
                    study = MO.run(number_of_trials=3)
                    best_values = study.best_trial.params

                    # ниже сохранение значений ф.п. при разных параметрах, а не лучших, лучшие сохраняются в transductive_results

                    columns = ["Loss name", "Conv", "l", "f", "cl", "asp", "ad", "loss value"] + list(
                        study.trials[0].params.keys())
                    df = pd.DataFrame(columns=columns)
                    for trial in study.trials:
                        row = [loss_name, conv, l, f, cl, asp, ad, trial.values[0]] + list(trial.params.values())
                        to_append = pd.Series(row, df.columns)
                        df = df.append(to_append, ignore_index=True)
                    df.to_csv('../data_help/' + str(loss_name) + '_' + str(conv) + '_' + str(name) + '.csv')

                    loss_trgt = {key: best_values[key] for key in loss if
                                 isinstance(loss[key], list) and key in best_values}
                    for par in loss:
                        if par not in loss_trgt:
                            loss_trgt[par] = loss[par]

                    M = Main(name=name, conv=conv, device=device, loss_function=loss_trgt, mode='unsupervised')
                    coherence, stable_rank, cond_number, self_cluster = M.run(best_values)
                    print('accuracies', coherence, stable_rank, cond_number, self_cluster)
                    to_append = pd.Series(
                        [loss_name, conv, f, cl, asp, ad, name, coherence, stable_rank, cond_number, self_cluster,
                         best_values], index=dataframe.columns)
                    dataframe = dataframe.append(to_append, ignore_index=True)
                    dataframe.to_csv('../results/transductive_results.csv')
