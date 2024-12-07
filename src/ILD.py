import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Função para calcular a diversidade intra-lista (ILD)
def compute_ILD(X_est, item_similarity, top_k=10):
    """
    Calcula a diversidade intra-lista (ILD) baseada na similaridade entre itens recomendados.
    """
    ild_per_user = []
    for _, user_row in X_est.iterrows():
        # Seleciona os itens recomendados com as maiores pontuações
        recommended_items = user_row.nlargest(top_k).index
        
        if len(recommended_items) > 1:
            # Calcula a similaridade média entre os itens recomendados
            pairwise_sim = [
                item_similarity.loc[i, j]
                for i in recommended_items
                for j in recommended_items
                if i != j
            ]
            ild = 1 - np.mean(pairwise_sim)  # Calcula a diversidade intra-lista
        else:
            ild = 0  # Diversidade nula se apenas 1 item for recomendado
        
        ild_per_user.append(ild)
    
    # Retorna a média da diversidade intra-lista para todos os usuários
    return np.mean(ild_per_user)


# Função para calcular as métricas por grupo
def compute_metrics_by_group(X, X_est, G, item_similarity, top_k=10):
    """
    Calcula a diversidade intra-lista (ILD) por grupo de usuários.
    """
    results = []
    for group_id, user_ids in G.items():
        # Seleciona os usuários do grupo
        X_group = X.loc[user_ids]
        X_est_group = X_est.loc[user_ids]

        # Calcula a diversidade intra-lista para o grupo
        ild = compute_ILD(X_est_group, item_similarity, top_k)

        # Armazena os resultados
        results.append({
            "group_id": group_id,
            "ILD": ild,
        })

    # Retorna um DataFrame com os resultados por grupo
    return pd.DataFrame(results)


# # Função para calcular a similaridade entre os itens
# def compute_item_similarity(X):
#     """
#     Calcula a similaridade entre os itens com base na matriz de avaliações.
#     """
#     # Substituir NaN por 0 para cálculos
#     X_filled = X.fillna(0)
#     # Calcular a matriz de similaridade utilizando a similaridade do coseno entre as colunas (itens)
#     item_similarity_matrix = cosine_similarity(X_filled.T)
#     return pd.DataFrame(item_similarity_matrix, index=X.columns, columns=X.columns)

# Função para calcular a similaridade entre os itens
def compute_item_similarity(X):
    """
    Calcula a similaridade entre os itens com base na matriz de avaliações.
    """
    # Calcular a matriz de similaridade utilizando correlação de Pearson entre as colunas (itens)
    item_similarity_matrix = compute_item_similarity_pearson(X)
    return item_similarity_matrix  


def compute_item_similarity_cosine(X):
    # Substituir NaN por 0 para cálculos
    X_filled = X.fillna(0)
    # Calcular a matriz de similaridade utilizando a similaridade do coseno entre as colunas (itens)
    item_similarity_matrix = cosine_similarity(X_filled.T)
    return pd.DataFrame(item_similarity_matrix, index=X.columns, columns=X.columns)

def compute_item_similarity_pearson(X):
    return X.corr(method='pearson')

def compute_item_similarity_jaccard(X):
    # Binário para estimar similaridade
    X_binary = X.notna().astype(int)
    return pd.DataFrame(
        cosine_similarity(X_binary.T),
        index=X.columns,
        columns=X.columns
    )

from sklearn.metrics import pairwise_distances

def compute_item_similarity_euclidean(X):
    distances = pairwise_distances(X.fillna(0).T, metric='euclidean')
    similarities = 1 / (1 + distances)  # Convertendo distância em similaridade
    return pd.DataFrame(similarities, index=X.columns, columns=X.columns)

from sklearn.metrics import pairwise_distances

def compute_item_similarity_minkowski(X, p=2):
    distances = pairwise_distances(X.fillna(0).T, metric='minkowski', p=p)
    similarities = 1 / (1 + distances)  # Convertendo distância em similaridade
    return pd.DataFrame(similarities, index=X.columns, columns=X.columns)

def compute_item_similarity_hamming(X):
    X_binary = X.notna().astype(int)
    return pd.DataFrame(
        cosine_similarity(X_binary.T),
        index=X.columns,
        columns=X.columns
    )

def compute_item_similarity_tanimoto(X):
    from sklearn.metrics import jaccard_score
    from itertools import combinations

    similarities = np.zeros((X.shape[1], X.shape[1]))
    for i, j in combinations(range(X.shape[1]), 2):
        similarities[i, j] = jaccard_score(X.iloc[:, i].notna().astype(int),
                                            X.iloc[:, j].notna().astype(int))
        similarities[j, i] = similarities[i, j]  # Matriz simétrica
    return pd.DataFrame(similarities, index=X.columns, columns=X.columns)