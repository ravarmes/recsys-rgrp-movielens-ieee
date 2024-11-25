import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

class RecSysSVD:
    def __init__(self, ratings):
        # Prepare the data for the Surprise library
        self.ratings = ratings
        self.reader = Reader(rating_scale=(1, 5))  # Ajuste o intervalo para as classificações que você está usando
        self.data = Dataset.load_from_df(ratings.reset_index(), self.reader)

        # Train-test split
        self.trainset, self.testset = train_test_split(self.data, test_size=0.2)

        # Initialize the SVD model
        self.model = SVD()
        
    def fit(self):
        # Fitting the model
        self.model.fit(self.trainset)
        
    def predict(self):
        # Make predictions
        predictions = self.model.test(self.testset)
        
        # Calculate RMSE
        rmse = accuracy.rmse(predictions)
        print(f'RMSE: {rmse}')

        return predictions

    def recommend(self, user_id, num_recommendations=5):
        """Recomendar itens para um usuário específico"""
        user_items = self.ratings[self.ratings['UserID'] == user_id]['ItemID']
        all_items = self.ratings['ItemID'].unique()
        items_to_predict = [item for item in all_items if item not in user_items.values]

        # Fazer previsões
        predictions = [self.model.predict(user_id, item) for item in items_to_predict]
        
        # Obter as melhores recomendações
        recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]
        return [(pred.uid, pred.iid, pred.est) for pred in recommendations]