from RecSys import RecSys
from UserFairness import Polarization
from UserFairness import IndividualLossVariance
from UserFairness import GroupLossVariance
import matplotlib.pyplot as plt
import novelty as novelty
import os
from contextlib import redirect_stdout

# Definindo o caminho do arquivo de saída
output_file = "output-gender.txt"

# Lendo dados de 3883 filmes e 6040 usuários
Data_path = 'Data/MovieLens-1M'
n_users= 1000
n_items= 1000
top_users = True  # True: usar usuários com mais classificações; False: caso contrário
top_items = True  # True: usar filmes com mais classificações; False: caso contrário

# Algoritmo de recomendação
algorithms = ['RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2', 'RecSysNCF2']

# Abrindo o arquivo para redirecionar saída
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

            # Identificando os grupos (NR: usuários agrupados pelo número de classificações disponíveis)
            list_users = X_est.index.tolist()
            masculine = users_info[users_info['Gender'] == 1].index.intersection(list_users).tolist()
            feminine = users_info[users_info['Gender'] == 2].index.intersection(list_users).tolist()

            G = {1: masculine, 2: feminine}

            glv = GroupLossVariance(X, omega, G, 1)
            RgrpGender = glv.evaluate(X_est)
            losses_RgrpGender = glv.get_losses(X_est)

            print("\n\n------------------------------------------")
            print(f'Algorithm: {algorithm}')
            print(f'Group (Rgrp Gender): {RgrpGender:.7f}')
            print(f'RgrpGender (masculine): {losses_RgrpGender[1]:.7f}')
            print(f'RgrpGender (feminine) : {losses_RgrpGender[2]:.7f}')

            # Calculando a popularidade e similaridade
            item_popularity = novelty.compute_item_popularity(X)
            item_similarity = novelty.compute_item_similarity(X)

            # Calculando métricas por grupo
            metrics_by_group = novelty.compute_metrics_by_group(X, X_est, G, item_popularity, item_similarity, top_k=50)
            print(metrics_by_group)

            # Adicione o bloco de código para gerar gráficos se necessário