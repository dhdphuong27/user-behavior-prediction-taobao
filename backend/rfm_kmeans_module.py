from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def prepare_rfm_and_elbow(df):
    """
    Performs RFM analysis and plots the Elbow method to help determine the optimal k.

    Args:
        df: A pandas DataFrame with user behavior data, including 'User_ID',
            'Datetime', and 'Behavior' (with 'Buy' actions).

    Returns:
        tuple: (rfm_df, wcss, elbow_fig) where:
            - rfm_df: A pandas DataFrame containing the RFM data with scaled values
            - wcss: List of WCSS values for the Elbow method
            - elbow_fig: The matplotlib figure object for the elbow plot
    """
    # Filter only 'Buy' actions for Monetary Value (M) - Although not used in clustering,
    # keeping this step for completeness if monetary value was available.
    purchases = df[df['Behavior'] == 'Buy'].copy()
    purchases['Datetime'] = pd.to_datetime(purchases['Datetime'])
    purchases['Date'] = purchases['Datetime'].dt.date

    # 1. Calculate recency
    recency = purchases.groupby(by='User_ID', as_index=False)['Date'].max()
    recency.columns = ['User_ID', 'LastPurchaseDate']
    current_date = purchases['Date'].max()

    recency['LastPurchaseDate'] = pd.to_datetime(recency['LastPurchaseDate'])
    current_date = pd.to_datetime(current_date)

    recency['Recency'] = recency['LastPurchaseDate'].apply(lambda x: (current_date - x).days)

    # 2. Calculate Frequency
    frequency = purchases.groupby('User_ID')['Behavior'].count().reset_index()
    frequency.columns = ['User_ID', 'Frequency']

    # 3. Create RFM table
    rfm = recency.merge(frequency, on='User_ID')
    rfm.drop(columns=['LastPurchaseDate'], inplace=True)

    # 4. Assign R, F quantile values
    quantiles = rfm.quantile(q=[0.25, 0.5, 0.75])
    quantiles = quantiles.to_dict()

    def r_score(x):
        if x <= quantiles['Recency'][0.25]:
            return 4
        elif x <= quantiles['Recency'][0.5]:
            return 3
        elif x <= quantiles['Recency'][0.75]:
            return 2
        else:
            return 1

    def f_score(x):
        if x <= quantiles['Frequency'][0.25]:
            return 1
        elif x <= quantiles['Frequency'][0.5]:
            return 2
        elif x <= quantiles['Frequency'][0.75]:
            return 3
        else:
            return 4

    rfm['R_score'] = rfm['Recency'].apply(r_score)
    rfm['F_score'] = rfm['Frequency'].apply(f_score)

    # Outlier handling using 95th percentile
    upper_limit_recency = rfm['Recency'].quantile(0.95)
    upper_limit_frequency = rfm['Frequency'].quantile(0.95)

    rfm['Recency'] = np.where(rfm['Recency'] > upper_limit_recency, upper_limit_recency, rfm['Recency'])
    rfm['Frequency'] = np.where(rfm['Frequency'] > upper_limit_frequency, upper_limit_frequency, rfm['Frequency'])

    # Scaling RFM values
    scaler = StandardScaler()
    scaler.fit(rfm[['Recency', 'Frequency']])
    rfm_scaled = scaler.transform(rfm[['Recency', 'Frequency']])
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=['RecencyScaled', 'FrequencyScaled'])
    rfm = pd.concat([rfm, rfm_scaled_df], axis=1)

    # Elbow Method for optimal k
    wcss = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(rfm[['RecencyScaled', 'FrequencyScaled']])
        wcss.append(kmeans.inertia_)

    # Create the Elbow Method graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(2, 11), wcss, marker='o')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    
    return rfm, wcss, fig

def perform_kmeans_clustering(rfm_df, optimal_k):
    """
    Performs K-means clustering with a specified number of clusters and adds
    cluster labels to the RFM DataFrame.

    Args:
        rfm_df: A pandas DataFrame containing the RFM data with scaled values
                (output from prepare_rfm_and_elbow function).
        optimal_k: The desired number of clusters for K-means.

    Returns:
        tuple: (rfm_df, cluster_means, customer_class, pie_fig) where:
            - rfm_df: A pandas DataFrame containing the original RFM data with the added 'Cluster' column
            - cluster_means: DataFrame with mean Recency and Frequency for each cluster
            - customer_class: DataFrame with cluster counts
            - pie_fig: The matplotlib figure object for the pie chart
    """
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_df[['RecencyScaled', 'FrequencyScaled']])
    
    cluster_means = rfm_df.groupby('Cluster')[['Recency', 'Frequency']].mean() 
    customer_class = rfm_df.groupby('Cluster')['User_ID'].count().reset_index()
    customer_class.columns = ['Customer Class', 'Counts']

    # Calculate the number of users in each cluster
    cluster_counts = rfm_df['Cluster'].value_counts().sort_index() # Sort by index to match cluster numbers

    # Map cluster indices to segment names for labels
    labels = [i+1 for i in range(optimal_k)]

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(cluster_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(cluster_counts)))
    ax.set_title('Distribution of Customers Across RFM Segments')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return rfm_df, cluster_means, customer_class, fig