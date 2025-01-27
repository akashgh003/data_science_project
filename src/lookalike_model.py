import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def create_customer_features(transactions_df, products_df):
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean'],
        'Quantity': ['sum', 'mean'],
        'TransactionID': 'count'
    }).reset_index()
    
    customer_metrics.columns = ['CustomerID', 'total_spend', 'avg_transaction',
                              'total_quantity', 'avg_quantity', 'transaction_count']
    
    merged_df = transactions_df.merge(products_df, on='ProductID')
    category_pivot = pd.pivot_table(
        merged_df,
        values='Quantity',
        index='CustomerID',
        columns='Category',
        aggfunc='sum',
        fill_value=0
    )
    
    return customer_metrics.merge(category_pivot, on='CustomerID', how='left')

def find_similar_customers(features_df, customer_id, n=3):
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(features_df.drop('CustomerID', axis=1))
    similarities = cosine_similarity(feature_matrix)
    customer_idx = features_df[features_df['CustomerID'] == customer_id].index[0]
    similar_indices = similarities[customer_idx].argsort()[::-1][1:n+1]
    
    return (features_df.iloc[similar_indices]['CustomerID'].values,
            similarities[customer_idx][similar_indices])

def generate_lookalikes(customers_df, products_df, transactions_df):
    features_df = create_customer_features(transactions_df, products_df)
    results = []
    
    for i in range(1, 21):
        customer_id = f'C{str(i).zfill(4)}'
        similar_customers, scores = find_similar_customers(features_df, customer_id)
        
        results.append({
            'customer_id': customer_id,
            'similar_1': similar_customers[0],
            'score_1': scores[0],
            'similar_2': similar_customers[1],
            'score_2': scores[1],
            'similar_3': similar_customers[2],
            'score_3': scores[2]
        })
    
    pd.DataFrame(results).to_csv('../outputs/Akash_Ghosh_Lookalike.csv', index=False)