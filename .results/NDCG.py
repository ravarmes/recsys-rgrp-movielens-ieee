import pandas as pd
import numpy as np

def group_ndcg_from_matrices(X, X_est, G, k=10):
    """
    Calcula o NDCG médio para diferentes grupos usando matrizes de avaliações e recomendações.
    
    Args:
        X (pd.DataFrame): Matriz de avaliações parcialmente preenchida.
                          Linhas representam IDs de usuários, colunas representam IDs ou títulos de itens.
        X_est (pd.DataFrame): Matriz de recomendações. 
                              Linhas são IDs de usuários e colunas são IDs ou títulos de itens recomendados.
        G (dict): Dicionário onde a chave é o ID do grupo e o valor é uma lista de IDs de usuários pertencentes ao grupo.
        k (int, optional): Número de itens no top-k para considerar. Padrão: 10.
        
    Returns:
        dict: NDCG médio por grupo.
    """
    def calculate_ndcg(relevance_scores, k):
        """Calcula o NDCG para uma lista de scores de relevância."""
        dcg = sum((rel / np.log2(idx + 2)) for idx, rel in enumerate(relevance_scores[:k]))
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        idcg = sum((rel / np.log2(idx + 2)) for idx, rel in enumerate(ideal_relevance_scores[:k]))
        return dcg / idcg if idcg > 0 else 0.0

    group_ndcg_scores = {}

    for group_id, user_ids in G.items():
        ndcg_scores = []

        for user_id in user_ids:
            if user_id in X.index and user_id in X_est.index:
                # Extrai os itens recomendados para o usuário
                user_recommendations = X_est.loc[user_id].sort_values(ascending=False).index.tolist()

                # Calcula as relevâncias dos itens recomendados a partir de X
                user_relevance = [
                    X.loc[user_id, item] if item in X.columns else np.nan
                    for item in user_recommendations
                ]

                # Substitui NaN por 0, assumindo itens não avaliados como irrelevantes
                user_relevance = [0 if np.isnan(rel) else rel for rel in user_relevance]

                # Calcula o NDCG para o usuário
                ndcg = calculate_ndcg(user_relevance, k)
                ndcg_scores.append(ndcg)

        # Calcula o NDCG médio do grupo
        group_ndcg_scores[group_id] = np.mean(ndcg_scores) if ndcg_scores else 0.0

    return group_ndcg_scores
