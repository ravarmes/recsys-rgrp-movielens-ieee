from RecSys import RecSys
from UserFairness import Polarization
from UserFairness import IndividualLossVariance
from UserFairness import GroupLossVariance
from UserFairness import RMSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
import novelty as novelty
import os
from contextlib import redirect_stdout

# Definindo o caminho do arquivo de saída
output_file = "output-agglomerative.txt"

# Lendo dados de 3883 filmes e 6040 usuários 
Data_path = 'Data/MovieLens-1M'
n_users = 1000
n_items = 1000
top_users = True  # True: usar usuários com mais classificações; False: caso contrário
top_items = True  # True: usar filmes com mais classificações; False: caso contrário

# Algoritmo de recomendação
# algorithms = ['RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2']
# algorithms = ['RecSysNCF', 'RecSysNCF', 'RecSysNCF', 'RecSysNCF', 'RecSysNCF', 'RecSysNCF', 'RecSysNCF', 'RecSysNCF', 'RecSysNCF', 'RecSysNCF']
algorithms = ['RecSysContentBased4'] * 50

resultados = []

# Abrindo o arquivo para redirecionar a saída
with open(output_file, 'w') as f:
    with redirect_stdout(f):
        for algorithm in algorithms:
            # Parâmetros para calcular medidas de fairness
            l = 5
            theta = 3
            k = 3

            recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

            X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir=Data_path)
            omega = ~X.isnull()  # matriz X com True nas células com avaliações e False nas células não avaliadas

            X_est = recsys.compute_X_est(X, algorithm)

            ilv = IndividualLossVariance(X, omega, 1)
            losses = ilv.get_losses(X_est)

            # Criando DataFrame
            df = pd.DataFrame(columns=['Gender', 'Age'])
            df['Gender'] = users_info['Gender']
            df['Age'] = users_info['Age']
            df['NR'] = users_info['NR']
            df['Loss'] = losses

            df.dropna(subset=['Loss'], inplace=True)  # eliminando linhas com valores vazios na coluna 'Loss'
            df = df.drop(columns=['Loss'])

            df_scaled = df.copy()
            df_scaled.iloc[:, :] = StandardScaler().fit_transform(df)

            Z = hierarchy.linkage(df_scaled, 'ward')

            n_clusters = 5
            cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            df_scaled['cluster_agglomerative'] = cluster.fit_predict(df_scaled)

            users = list(df_scaled.index)
            groups = df_scaled['cluster_agglomerative']

            grouped_users = {i: [] for i in range(n_clusters)}
            for user, group in zip(users, groups):
                grouped_users[group].append(user)

            G = {1: grouped_users[0], 2: grouped_users[1], 3: grouped_users[2], 4: grouped_users[3], 5: grouped_users[4]}

            # Calculando a quantidade de elementos em cada grupo
            quantidades = {key: len(value) for key, value in G.items()}
            for key, quantidade in quantidades.items():
                print(f"Grupo {key}: {quantidade} elementos")

            glv = GroupLossVariance(X, omega, G, 1)
            RgrpAgglomerative = glv.evaluate(X_est)
            losses_RgrpAgglomerative = glv.get_losses(X_est)

            print("\n\n------------------------------------------")
            print(f'Algorithm: {algorithm}')
            print(f'Group (Rgrp Agglomerative): {RgrpAgglomerative:.7f}')
            print(f'RgrpAgglomerative (1): {losses_RgrpAgglomerative[1]:.7f}')
            print(f'RgrpAgglomerative (2): {losses_RgrpAgglomerative[2]:.7f}')
            print(f'RgrpAgglomerative (3): {losses_RgrpAgglomerative[3]:.7f}')
            print(f'RgrpAgglomerative (4): {losses_RgrpAgglomerative[4]:.7f}')
            print(f'RgrpAgglomerative (5): {losses_RgrpAgglomerative[5]:.7f}')

            # Gerando o gráfico
            RgrpAgglomerative_groups = ['G1', 'G2', 'G3', 'G4', 'G5']
            plt.bar(RgrpAgglomerative_groups, losses_RgrpAgglomerative)
            plt.title(f'Rgrp (Aglomerativo) ({algorithm}): {RgrpAgglomerative:.7f}')
            plt.savefig(f'plots/RgrpAgglomerative-{algorithm}.png')  # Salva o gráfico como PNG
            plt.clf()  # Limpa o gráfico para o próximo

            # Calculando a popularidade e similaridade
            item_popularity = novelty.compute_item_popularity(X)
            item_similarity = novelty.compute_item_similarity(X)

            # Calculando métricas por grupo
            metrics_by_group = novelty.compute_metrics_by_group(X, X_est, G, item_popularity, item_similarity, top_k=50)
            print(metrics_by_group)

            # Calculando RMSE
            rmse = RMSE(X, omega)
            rmse_result = rmse.evaluate(X_est)
            print(f'RMSE: {rmse_result:.7f}')

            # Armazenando os resultados para cada algoritmo
            resultados.append(f'{RgrpAgglomerative:.7f}')

print(resultados)  # Exibe todos os resultados
menor = min(resultados)
print(menor) # Exibe o menor dos resultados