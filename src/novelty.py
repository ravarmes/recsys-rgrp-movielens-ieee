import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Função para calcular a popularidade dos itens
def compute_item_popularity(X):
    """
    Calcula a popularidade dos itens como a proporção de usuários que avaliaram o item.
    """
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
        
        # Verifica se algum item recomendado tem popularidade 0 e ajusta
        novelty_values = -np.log(np.clip(item_popularity[recommended_items] + epsilon, epsilon, 1))
        
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

# Função para calcular as métricas por grupo
def compute_metrics_by_group(X, X_est, G, item_popularity, item_similarity, top_k=10):
    """
    Calcula novidade e diversidade por grupo de usuários.
    """
    results = []
    for group_id, user_ids in G.items():
        # print(user_ids)
        # Seleciona os usuários do grupo
        X_group = X.loc[user_ids]
        X_est_group = X_est.loc[user_ids]
        # print("X_est_group")
        # print(X_est_group)

        # Calcula as métricas para o grupo
        novelty = compute_novelty(X_est_group, item_popularity, top_k)
        ild = compute_ILD(X_est_group, item_similarity, top_k)
        adjusted_diversity = compute_adjusted_diversity(ild, novelty)

        # Armazena os resultados
        results.append({
            "group_id": group_id,
            "novelty": novelty,
            "ILD": ild,
            "adjusted_diversity": adjusted_diversity
        })

    return pd.DataFrame(results)

# # Exemplo de uso
# if __name__ == "__main__":
#     np.random.seed(42)

#     # Simulação de matrizes X e X_est
#     num_users, num_items = 300, 1000
#     X = pd.DataFrame(
#         np.random.rand(num_users, num_items) * (np.random.rand(num_users, num_items) > 0.8),
#         columns=[f"item_{i}" for i in range(num_items)]
#     )
#     X_est = pd.DataFrame(
#         np.random.rand(num_users, num_items),
#         columns=[f"item_{i}" for i in range(num_items)]
#     )

#     # Configuração do dicionário de grupos
#     G = {
#         1: list(range(0, 150)),  # Grupo 1: usuários 0 a 149
#         2: list(range(150, 300))  # Grupo 2: usuários 150 a 299
#     }

#     # Calculando a popularidade e similaridade
#     item_popularity = compute_item_popularity(X)
#     item_similarity = compute_item_similarity(X)

#     # Calculando métricas por grupo
#     metrics_by_group = compute_metrics_by_group(X, X_est, G, item_popularity, item_similarity, top_k=10)
#     print(metrics_by_group)
