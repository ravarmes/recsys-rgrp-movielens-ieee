from RecSys import RecSys
import pandas as pd

# Caminho para os dados
Data_path = 'Data/MovieLens-1M'
n_users = 1000
n_items = 1000
top_users = True
top_items = True

# Inicializar RecSys
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

# Obter a matriz X
X, users_info, items_info = recsys.read_movielens_1M(
    n_users, n_items, top_users, top_items, data_dir=Data_path
)

# Verificar dimensões da matriz X
print(f"Dimensões de X: {X.shape}")  # Deve ser (1000, 1000)

# Caminho para o arquivo movies.dat
movies_file = f"{Data_path}/movies.dat"

# Carregar dados do arquivo movies.dat
movies = pd.read_csv(
    movies_file,
    sep='::',
    header=None,
    engine='python',
    names=['MovieID', 'Title', 'Genres'],
    encoding='latin1'
)

# Obter os títulos dos filmes de X e fazer a limpeza
selected_movie_titles = X.columns.str.strip().tolist()  # Remover espaços em branco

# Depuração: imprimir os títulos dos filmes em X
print("Títulos dos filmes na matriz X:")
for title in selected_movie_titles:
    print(title)

# Remover espaços em branco do título no DataFrame movies
movies['Title'] = movies['Title'].str.strip()  # Limpar espaços em branco nos títulos

# Filtrar os filmes correspondentes aos títulos no DataFrame X
filtered_movies = movies[movies['Title'].isin(selected_movie_titles)]

# Depuração: imprimir os filmes que foram filtrados
print("\nFilmes encontrados no arquivo movies.dat que correspondem a X:")
print(filtered_movies)

# Certificar de que estamos limitando a 1000 filmes
if len(filtered_movies) > n_items:
    filtered_movies = filtered_movies.head(n_items)

# Criar o arquivo TXT com títulos e gêneros
output_file = "selected_movies.txt"

with open(output_file, "w") as f:
    for _, row in filtered_movies.iterrows():
        f.write(f"{row['Title']}::{row['Genres']}\n")

print(f"Arquivo '{output_file}' gerado com sucesso! Apenas os filmes do dataframe X foram incluídos.")