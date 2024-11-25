from itertools import product
import RecSysNCF2
from RecSys import RecSys

# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users=  1000
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use movies with more ratings; False: otherwise

# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

ratings_df, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
omega = ~ratings_df.isnull() # matrix X with True in cells with evaluations and False in cells not rated

# Defina as listas de valores para cada hiperparâmetro
n_factors_list = [10, 20, 50, 100]
epochs_list = [10, 20, 30]
batch_size_list = [16, 32, 64]
learning_rate_list = [0.001, 0.01, 0.05]

best_error = float('inf')
best_params = None
best_model = None

# Testa todas as combinações de hiperparâmetros
for n_factors, epochs, batch_size, learning_rate in product(n_factors_list, epochs_list, batch_size_list, learning_rate_list):
    # Criação e ajuste do modelo
    model = RecSysNCF2.RecSysNCF2(n_users=ratings_df.shape[0], n_items=ratings_df.shape[1], n_factors=n_factors, ratings=ratings_df)
    _, error = model.fit_model(epochs=epochs, batch_size=batch_size)

    # Compare e armazene o melhor modelo
    if error < best_error:
        best_error = error
        best_params = (n_factors, epochs, batch_size, learning_rate)
        best_model = model

print(f"Melhores parâmetros: n_factors={best_params[0]}, epochs={best_params[1]}, batch_size={best_params[2]}, learning_rate={best_params[3]}")
print(f"Melhor RMSE: {best_error}")