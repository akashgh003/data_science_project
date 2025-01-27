import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

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

def find_optimal_clusters(features, output_path, max_clusters=10):
    feature_matrix = features.drop('CustomerID', axis=1)
    db_scores = []
    silhouette_scores = []
    
    for n in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        labels = kmeans.fit_predict(feature_matrix)
        db_scores.append(davies_bouldin_score(feature_matrix, labels))
        silhouette_scores.append(silhouette_score(feature_matrix, labels))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), db_scores, marker='o')
    plt.title('Davies-Bouldin Score vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'clustering_metrics.png'))
    plt.close()
    
    return np.argmin(db_scores) + 2

def generate_clustering_report(n_clusters, db_score, silhouette, cluster_stats, output_path):
    pdf_path = os.path.join(output_path, 'Akash_Ghosh_Clustering.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Customer Segmentation Analysis", title_style))
    story.append(Spacer(1, 20))
    
    # Metrics
    story.append(Paragraph("Clustering Metrics", styles['Heading2']))
    story.append(Paragraph(f"Number of Clusters: {n_clusters}", styles['Normal']))
    story.append(Paragraph(f"Davies-Bouldin Index: {db_score:.4f}", styles['Normal']))
    story.append(Paragraph(f"Silhouette Score: {silhouette:.4f}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Cluster Statistics
    story.append(Paragraph("Cluster Statistics", styles['Heading2']))
    for cluster, stats in cluster_stats.items():
        story.append(Paragraph(f"Cluster {cluster}:", styles['Heading3']))
        story.append(Paragraph(f"Size: {stats['size']} customers", styles['Normal']))
        story.append(Paragraph(f"Average Spend: ${stats['avg_spend']:.2f}", styles['Normal']))
        story.append(Paragraph(f"Average Transactions: {stats['avg_transactions']:.1f}", styles['Normal']))
        story.append(Spacer(1, 10))
    
    doc.build(story)

def perform_clustering(customers_df, products_df, transactions_df):
    try:
        output_path = os.environ.get('PROJECT_OUTPUT_PATH', 'outputs')
        os.makedirs(output_path, exist_ok=True)
        
        features_df = prepare_features(customers_df, transactions_df)
        feature_cols = [col for col in features_df.columns if col != 'CustomerID']
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df[feature_cols])
        
        n_clusters = find_optimal_clusters(features_df, output_path)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        db_score = davies_bouldin_score(scaled_features, cluster_labels)
        silhouette = silhouette_score(scaled_features, cluster_labels)
        
        # Calculate cluster statistics
        features_df['Cluster'] = cluster_labels
        cluster_stats = {}
        for cluster in range(n_clusters):
            cluster_data = features_df[features_df['Cluster'] == cluster]
            cluster_stats[cluster] = {
                'size': len(cluster_data),
                'avg_spend': cluster_data['total_spend'].mean(),
                'avg_transactions': cluster_data['transaction_count'].mean()
            }
        
        generate_clustering_report(n_clusters, db_score, silhouette, cluster_stats, output_path)
        
        results_df = pd.DataFrame({
            'CustomerID': features_df['CustomerID'],
            'Cluster': cluster_labels
        })
        results_df.to_csv(os.path.join(output_path, 'clustering_results.csv'), index=False)
        
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        raise