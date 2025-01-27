import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def prepare_features(customers_df, transactions_df):
    transaction_features = transactions_df.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean', 'std'],
        'Quantity': ['sum', 'mean'],
        'TransactionID': ['count']
    }).reset_index()
    
    transaction_features.columns = ['CustomerID', 'total_spend', 'avg_transaction', 
                                  'std_transaction', 'total_quantity', 'avg_quantity',
                                  'transaction_count']
    
    transaction_features['std_transaction'] = transaction_features['std_transaction'].fillna(0)
    
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    customers_df['account_age'] = (pd.Timestamp.now() - customers_df['SignupDate']).dt.days
    
    features = transaction_features.merge(
        customers_df[['CustomerID', 'Region', 'account_age']],
        on='CustomerID'
    )
    
    features = pd.get_dummies(features, columns=['Region'])
    feature_cols = [col for col in features.columns if col != 'CustomerID']
    
    imputer = SimpleImputer(strategy='median')
    features_imputed = imputer.fit_transform(features[feature_cols])
    
    features_clean = pd.DataFrame(features_imputed, columns=feature_cols)
    features_clean['CustomerID'] = features['CustomerID']
    
    return features_clean

def find_optimal_clusters(features, max_clusters=10):
    feature_matrix = features.drop('CustomerID', axis=1)
    db_scores = []
    silhouette_scores = []
    
    for n in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        labels = kmeans.fit_predict(feature_matrix)
        db_scores.append(davies_bouldin_score(feature_matrix, labels))
        silhouette_scores.append(silhouette_score(feature_matrix, labels))
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_clusters + 1), db_scores, marker='o')
    plt.title('Davies-Bouldin Score vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score')
    plt.savefig('../outputs/clustering_metrics.png')
    plt.close()
    
    return np.argmin(db_scores) + 2

def perform_clustering(customers_df, products_df, transactions_df):
    try:
        features_df = prepare_features(customers_df, transactions_df)
        feature_cols = [col for col in features_df.columns if col != 'CustomerID']
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df[feature_cols])
        n_clusters = find_optimal_clusters(features_df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        db_score = davies_bouldin_score(scaled_features, cluster_labels)
        silhouette = silhouette_score(scaled_features, cluster_labels)
        
        results_df = pd.DataFrame({
            'CustomerID': features_df['CustomerID'],
            'Cluster': cluster_labels
        })
        
        with open('../outputs/Akash_Ghosh_Clustering.pdf', 'w') as f:
            f.write(f"Number of clusters: {n_clusters}\n")
            f.write(f"Davies-Bouldin Index: {db_score:.4f}\n")
            f.write(f"Silhouette Score: {silhouette:.4f}\n")
            
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            f.write("\nCluster Sizes:\n")
            for cluster, size in cluster_sizes.items():
                f.write(f"Cluster {cluster}: {size} customers\n")
        
        results_df.to_csv('../outputs/clustering_results.csv', index=False)
        
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        raise