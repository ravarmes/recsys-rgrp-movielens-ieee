import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from abc import ABCMeta, abstractmethod
from sklearn.metrics import jaccard_score

class RecSysContentBased2:
    __metaclass__ = ABCMeta
    
    def __init__(self, k, ratings=None, user_based=True, movie_file=None):
        self.k = k
        self.user_based = user_based
        self.movie_file = movie_file
        if ratings is not None:
            self.set_ratings(ratings)
    
    def set_ratings(self, ratings):
        self.ratings = ratings
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_item = (~self.ratings.isnull()).sum(axis=0)
    

    def load_movie_genres(self):
        """
        Carrega os gêneros dos filmes do arquivo fornecido e cria uma matriz de características binárias.
        """
        # Carregar o arquivo de filmes
        movies = pd.read_csv(
            self.movie_file,
            sep='::',
            header=None,
            names=['Title', 'Genres'],
            engine='python',
            encoding='latin1'
        )
        
        # Criar uma lista de listas de gêneros
        movies['Genres'] = movies['Genres'].str.split('|')

        # Imprimir as primeiras linhas do DataFrame de filmes para verificação
        print("DataFrame de filmes:")
        print(movies.head())

        # Criar uma lista única de gêneros
        genres = list(set(genre for sublist in movies['Genres'] for genre in sublist))
        
        # Criar a matriz de características binárias dos gêneros
        genre_matrix = pd.DataFrame(0, index=movies['Title'], columns=genres)
        
        for _, row in movies.iterrows():
            genre_matrix.loc[row['Title'], row['Genres']] = 1  # Marcar os gêneros correspondentes

        # Imprimir a matriz de gêneros
        print("Matriz de gêneros:")
        print(genre_matrix.head())  # Mostra as primeiras linhas da matriz

        # Verificar a soma dos gêneros para cada coluna
        genre_sum = genre_matrix.sum(axis=0)
        print("Soma de gêneros por filme:")
        print(genre_sum[genre_sum > 0])  # Mostra apenas gêneros que têm pelo menos uma ocorrência

        self.genre_matrix = genre_matrix
    
    def get_similarity_matrix(self):
        """
        Calcula a matriz de similaridade baseada nos gêneros dos filmes.
        """
        if self.genre_matrix is None:
            raise ValueError("A matriz de gêneros não foi carregada.")
        
        # Calcular a similaridade utilizando o cosseno
        similarity = pd.DataFrame(
            cosine_similarity(self.genre_matrix),
            index=self.genre_matrix.index,
            columns=self.genre_matrix.index
        )
        
        # Verificar uma amostra da matriz de similaridade
        print("Matriz de similaridade (exemplo):")
        print(similarity.head())

        return similarity
    

    def get_jaccard_similarity(self):
        """
        Calcula a matriz de similaridade utilizando o índice de Jaccard de forma otimizada.
        """
        if self.genre_matrix is None:
            raise ValueError("A matriz de gêneros não foi carregada.")
        
        # Obtém a matriz binária
        binary_matrix = self.genre_matrix.values.astype(np.float32)  # Converter para float32 para economizar memória e CPU
        
        # Cálculo de interseção e união
        intersection = np.dot(binary_matrix.astype(bool).astype(int), binary_matrix.T.astype(bool).astype(int))
        union = binary_matrix.sum(axis=1).reshape(-1, 1) + binary_matrix.sum(axis=1) - intersection

        # Para evitar divisão por zero
        union[union == 0] = 1  # Substitui zeros para evitar divisão por zero

        # Cálculo da média de Jaccard
        jaccard_similarity = intersection / union
        
        # Cria um DataFrame para fácil leitura
        similarity_df = pd.DataFrame(jaccard_similarity, index=self.genre_matrix.index, columns=self.genre_matrix.index)
        
        print("Matriz de similaridade de Jaccard (exemplo):")
        print(similarity_df.head())  # Mostra as primeiras linhas da matriz de similaridade

        return similarity_df

    def knn_filtering(self, similarity):
        """
        Realiza o filtro de k-vizinhos mais próximos, zerando os filmes menos similares.
        """
        for i in similarity.index:
            sorted_neighbors = similarity.loc[i].sort_values(ascending=False)
            keep = sorted_neighbors.iloc[:self.k+1].index
            similarity.loc[i, ~similarity.columns.isin(keep)] = 0
        similarity = similarity.div(similarity.sum(axis=1), axis=0)
        return similarity

    
    def fit_model(self, max_iter=200, threshold=1e-5):
        if self.movie_file is None:
            raise ValueError("O arquivo de filmes não foi fornecido.")

        self.load_movie_genres()  # Carrega as informações dos gêneros dos filmes
        # similarity = self.get_similarity_matrix()  # Obtém a matriz de similaridade
        similarity = self.get_similarity_matrix()  # Obtém a matriz de similaridade

        # Ou descomente a linha abaixo para usar Jaccard
        # similarity = self.get_jaccard_similarity()  # Calcula a matriz de similaridade usando Jaccard

        knn_similarity = self.knn_filtering(similarity)  # Aplica o filtro KNN

        print(f"Dimensões da matriz de ratings: {self.ratings.shape}")

        knn_similarity = knn_similarity.reindex(columns=self.ratings.columns, index=self.ratings.columns)

        pred_raw = self.ratings.fillna(0).dot(knn_similarity)  # Predição sem normalização
        print(f"Dimensões da matriz pred (raw): {pred_raw.shape}")

        pred = pred_raw.copy()
        sum_similarities = knn_similarity.sum(axis=0)

        for i in range(len(pred.columns)):
            if sum_similarities[i] > 0:
                pred.iloc[:, i] = pred.iloc[:, i] / sum_similarities[i]

        # Clipping para garantir que os valores estejam entre 1 e 5
        pred = pred.clip(lower=1, upper=5)

        # Aplicando ajuste com base na média de avaliações do usuário
        # Ajusta a predição com base na média de notas de cada filme
        avg_ratings = self.ratings.mean(axis=0, skipna=True)  # Média das classificações por filme
        for i in range(len(pred.columns)):
            if avg_ratings[i] > 0:
                pred.iloc[:, i] = (pred.iloc[:, i] + avg_ratings[i]) / 2  # Combine a predição com a média

        pred = pred.reindex(columns=self.ratings.columns)

        self.pred = pred.fillna(0)  # Preenche qualquer NaN restante com 0
        self.U = self.ratings
        self.V = knn_similarity

        return self.pred  # Retorna a matriz de predições