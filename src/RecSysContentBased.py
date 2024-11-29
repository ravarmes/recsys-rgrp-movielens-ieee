import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


class RecSysContentBased:
    def __init__(self, k, ratings, movies_file):
        """
        Inicializa o sistema de recomendação baseado em conteúdo.

        Args:
            k (int): Número de vizinhos mais próximos.
            ratings (pd.DataFrame): Matriz de avaliações (usuários x filmes).
            movies_file (str): Caminho para o arquivo movies.dat.
        """
        self.k = k
        self.ratings = ratings
        self.movies_file = movies_file
        self.movies = None
        self.movie_features = None
        self.set_movies()

    def set_movies(self):
        """
        Carrega o arquivo movies.dat, mapeia os títulos dos filmes em self.ratings
        com os filmes em movies.dat e identifica os gêneros.
        """
        try:
            # Carregar o arquivo movies.dat
            self.movies = pd.read_csv(
                self.movies_file,
                sep='::',
                header=None,
                names=['MovieID', 'Title', 'Genres'],
                engine='python',
                encoding='latin-1'  # Alteração aqui
            )

            # Mapear títulos de filmes em self.ratings para movies.dat
            ratings_titles = self.ratings.columns.tolist()
            mapped_movies = self.movies[self.movies['Title'].isin(ratings_titles)]

            if mapped_movies.empty:
                raise ValueError("Nenhum ID de filme em 'self.ratings.columns' corresponde a 'movies.dat'.")

            # Criar matriz de gêneros binários
            vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
            self.movie_features = vectorizer.fit_transform(mapped_movies['Genres'])

        except Exception as e:
            print(f"Erro ao configurar os filmes: {e}")
            raise


    def get_similarity_matrix(self):
        """
        Calcula a matriz de similaridade entre os filmes com base nos gêneros.

        Returns:
            np.array: Matriz de similaridade baseada em cosseno.
        """
        if self.movie_features is None:
            raise ValueError("As características dos filmes ainda não foram definidas.")
        return cosine_similarity(self.movie_features)

    def fit_model(self):
        """
        Calcula as recomendações usando um modelo de filtragem baseada em conteúdo.
        """
        try:
            # Matriz de similaridade KNN (1000x1000)
            knn_similarity = cosine_similarity(self.movie_features, self.movie_features)
            
            # Converter para DataFrame para alinhar índices e colunas
            knn_similarity = pd.DataFrame(
                knn_similarity, 
                index=self.ratings.columns, 
                columns=self.ratings.columns
            )
            
            # Alinhar os índices/colunas de `knn_similarity` com as colunas de `self.ratings`
            knn_similarity = knn_similarity.loc[self.ratings.columns, self.ratings.columns]

            # Predições normalizadas
            pred = knn_similarity.dot(self.ratings.fillna(0))
            pred = pred.div(knn_similarity.sum(axis=1).replace(0, np.nan), axis=0)
            
            # Substituir NaN por 0 (se necessário)
            pred = pred.fillna(0)

            return pred

        except Exception as e:
            print(f"Erro inesperado: {e}")
            raise



    def recommend_for_user(self, user_id, n_recommendations=10):
        """
        Gera recomendações para um usuário específico.

        Args:
            user_id (int): ID do usuário.
            n_recommendations (int): Número de recomendações a serem geradas.

        Returns:
            list: Lista de títulos de filmes recomendados.
        """
        try:
            # Obter as avaliações previstas
            predictions = self.fit_model()

            # Selecionar filmes não avaliados pelo usuário
            user_ratings = self.ratings.loc[user_id]
            non_watched = user_ratings[user_ratings.isna()].index

            # Ordenar por previsão e recomendar
            recommendations = predictions.loc[user_id, non_watched].sort_values(ascending=False)
            return recommendations.head(n_recommendations).index.tolist()
        except Exception as e:
            print(f"Erro ao gerar recomendações para o usuário {user_id}: {e}")
            raise
