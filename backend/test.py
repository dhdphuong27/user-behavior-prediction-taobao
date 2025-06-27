import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import random

def recommendations(model, target_product):
    return model.recommend(target_product, n=5)

class ProductRecommender:
    def __init__(self, min_user_interactions=5, min_product_interactions=10):
        self.min_user_interactions = min_user_interactions
        self.min_product_interactions = min_product_interactions
        self.item_similarity = None
        self.product_encoder = None

    def fit(self, df):
        user_counts = df['User_ID'].value_counts()
        product_counts = df['Product_ID'].value_counts()

        filtered_df = df[
            df['User_ID'].isin(user_counts[user_counts >= self.min_user_interactions].index) &
            df['Product_ID'].isin(product_counts[product_counts >= self.min_product_interactions].index)
        ]

        user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()

        user_labels = user_encoder.fit_transform(filtered_df['User_ID'])
        product_labels = self.product_encoder.fit_transform(filtered_df['Product_ID'])

        interaction_matrix = coo_matrix(
            (np.ones(len(filtered_df)), (user_labels, product_labels))
        ).tocsr()

        item_matrix = interaction_matrix.T
        self.item_similarity = cosine_similarity(item_matrix)
        return filtered_df

    def recommend(self, product_id, n=5):
        if self.product_encoder is None or self.item_similarity is None:
            raise Exception("Model not trained. Call fit() first.")
        if product_id not in self.product_encoder.classes_:
            return f"Product {product_id} not found in the filtered data."

        idx = self.product_encoder.transform([product_id])[0]
        sim_scores = list(enumerate(self.item_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices_scores = [(i, score) for i, score in sim_scores[1:n+1]]

        recommendations = [
            {'Product_ID': self.product_encoder.inverse_transform([i])[0], 'Score': round(score, 4)}
            for i, score in top_indices_scores
        ]
        return recommendations



col_names = ["User_ID", "Product_ID", "Category_ID", "Behavior", "Timestamp"]
col_use = ["User_ID","Product_ID","Behavior"]
df = pd.read_csv("UserBehavior.csv", names=col_names, usecols=col_use)
df = df[df['Behavior'] == 'buy'] #can't handle 100mil
df = df.drop('Behavior', axis=1)

model = ProductRecommender(min_product_interactions=5)

viable_products_recomendations = model.fit(df)

print(recommendations(model, random.choice(viable_products_recomendations))) 




