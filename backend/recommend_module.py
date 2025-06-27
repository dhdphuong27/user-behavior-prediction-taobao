# from surprise import SVD, Dataset, Reader
# from surprise.model_selection import train_test_split
# from surprise import accuracy
# import pandas as pd
# from prophet import Prophet
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load Taobao data
# data = df
# behavior_weights = {'PageView': 1, 'Favorite': 3, 'AddToCart': 5, 'Buy': 10}
# data['Rating'] = data['Behavior'].map(behavior_weights)
# data_sample = data.sample(n=1000000, random_state=42)

# # Prepare for Surprise
# reader = Reader(rating_scale=(1, 10))
# dataset = Dataset.load_from_df(data_sample[['User_ID', 'Product_ID', 'Rating']], reader)
# trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# # Train and evaluate SVD
# model = SVD(n_factors=50, n_epochs=20, random_state=42)
# model.fit(trainset)
# predictions = model.test(testset)
# rmse = accuracy.rmse(predictions)

# # Predict for a user
# user_id = 12345
# items = data_sample['Product_ID'].unique()
# top_n = sorted([(item, model.predict(user_id, item).est) for item in items], key=lambda x: x[1], reverse=True)[:10]
# print(f'Top 10 recommendations: {top_n}')

#Since surprise doedn't work for now, temporarily returning top 10 products as placeholder

import pandas as pd
import random 
col_names = ["User_ID", "Product_ID", "Category_ID", "Behavior", "Timestamp"]
col = ['User_ID']
df = pd.read_csv('csv_chunks/chunk_01.csv', names=col_names, usecols=col)

items = df['User_ID'].value_counts().head(10)

def recommend_products(user_id, num_recommendations=10):
    # Temporarily returning top 10 products
    return items