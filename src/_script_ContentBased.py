import pandas as pd
import numpy as np
from RecSys import RecSys
from UserFairness import RMSE
import novelty
import RecSysContentBased3

# Definir parâmetros para a busca
k_values = [50, 75, 90, 100, 110, 120]  # Exemplo de valores para o parâmetro k
reg_values = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # Valores para o parâmetro de regularização
best_rmse = float('inf')
best_params = None

# Configurações de dados
Data_path = 'Data/MovieLens-1M'
n_users = 1000
n_items = 1000
top_users = True
top_items = True

# Lendo dados
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)
X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir=Data_path)
omega = ~X.isnull()  # Matriz X com True nas células com avaliações e False onde não foram avaliadas.

# Testar cada combinação de k e regularization
for k in k_values:
    for reg in reg_values:
        print(f'Testing k={k}, regularization={reg}')
        algorithm = 'RecSysContentBased3'
        
        RS = RecSysContentBased3.RecSysContentBased3(k=k, ratings=X, movie_file='Data/MovieLens-1M/movies-1000.txt', regularization=reg)
        X_est = RS.fit_model()

        # calcular RMSE
        rmse = RMSE(X, omega)
        rmse_result = rmse.evaluate(X_est)

        print(f'RMSE for k={k}, regularization={reg}: {rmse_result:.7f}')

        # Verifica se o RMSE atual é o melhor encontrado
        if rmse_result < best_rmse:
            best_rmse = rmse_result
            best_params = (k, reg)

print(f'\nBest RMSE: {best_rmse:.7f} with parameters k={best_params[0]} and regularization={best_params[1]}')