import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Função para calcular a popularidade dos itens
def compute_item_popularity(X):
    """
    Calcula a popularidade dos itens como a proporção de usuários que avaliaram o item.
    """
    # Substitui NaN por 0 para facilitar a contagem
    X_filled = X.notna().astype(int)
    item_counts = X_filled.sum(axis=0)  # Soma por coluna
    popularity = item_counts / len(X)  # Proporção de usuários que avaliaram cada item
    return popularity

# Função para calcular a novidade
def compute_novelty(X_est, item_popularity, top_k=10, epsilon=1e-6):
    """
    Calcula a novidade global com base na popularidade dos itens recomendados.
    """
    novelty_per_user = []
    for _, user_row in X_est.iterrows():
        # Seleciona os itens recomendados com as maiores pontuações
        recommended_items = user_row.nlargest(top_k).index
        novelty_values = -np.log(item_popularity[recommended_items] + epsilon)
        novelty_per_user.append(novelty_values.mean())
    return np.mean(novelty_per_user)

# Função para calcular a diversidade intra-lista (ILD)
def compute_ILD(X_est, item_similarity, top_k=10):
    """
    Calcula a diversidade intra-lista (ILD) baseada na similaridade entre itens recomendados.
    """
    ild_per_user = []
    for _, user_row in X_est.iterrows():
        recommended_items = user_row.nlargest(top_k).index
        if len(recommended_items) > 1:
            # Similaridade média entre os itens recomendados
            pairwise_sim = [
                item_similarity.loc[i, j]
                for i in recommended_items
                for j in recommended_items
                if i != j
            ]
            ild = 1 - np.mean(pairwise_sim)  # Diversidade intra-lista
        else:
            ild = 0  # Diversidade nula se apenas 1 item for recomendado
        ild_per_user.append(ild)
    return np.mean(ild_per_user)

# Função para calcular a similaridade entre os itens
def compute_item_similarity(X):
    """
    Calcula a similaridade entre os itens com base na matriz de avaliações.
    """
    X_filled = X.fillna(0)  # Substituir NaN por 0 para cálculos
    item_similarity_matrix = cosine_similarity(X_filled.T)  # Similaridade entre colunas (itens)
    return pd.DataFrame(item_similarity_matrix, index=X.columns, columns=X.columns)

# Função para calcular a diversidade ajustada
def compute_adjusted_diversity(ild, novelty, epsilon=1e-6):
    """
    Calcula a diversidade ajustada como o produto da diversidade intra-lista e a novidade.
    """
    return ild * (novelty + epsilon)

# # Geração de dados simulados
# np.random.seed(42)
# num_users, num_items = 300, 1000
# X = pd.DataFrame(
#     np.random.rand(num_users, num_items) * (np.random.rand(num_users, num_items) > 0.8),
#     columns=[f"item_{i}" for i in range(num_items)]
# )
# X_est = pd.DataFrame(
#     np.random.rand(num_users, num_items),
#     columns=[f"item_{i}" for i in range(num_items)]
# )

# # Passo 1: Calcular a popularidade dos itens
# item_popularity = compute_item_popularity(X)

# # Passo 2: Calcular a similaridade entre os itens
# item_similarity = compute_item_similarity(X)

# # Passo 3: Calcular a novidade
# novelty_score = compute_novelty(X_est, item_popularity)
# print(f"Novidade Global: {novelty_score:.4f}")

# # Passo 4: Calcular a diversidade intra-lista
# ild_score = compute_ILD(X_est, item_similarity)
# print(f"Diversidade Intra-Lista (ILD): {ild_score:.4f}")

# # Passo 5: Calcular a diversidade ajustada
# diversity_adjusted = compute_adjusted_diversity(ild_score, novelty_score)
# print(f"Diversidade Ajustada: {diversity_adjusted:.4f}")
