import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def analyze_customers(customers_df):
    plt.figure(figsize=(12, 5))
    customers_df['Region'].value_counts().plot(kind='bar')
    plt.title('Customer Distribution by Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../outputs/customer_analysis.png')
    plt.close()

def analyze_products(products_df):
    plt.figure(figsize=(12, 5))
    sns.histplot(products_df['Price'])
    plt.title('Product Price Distribution')
    plt.tight_layout()
    plt.savefig('../outputs/product_analysis.png')
    plt.close()

def analyze_transactions(transactions_df, products_df):
    merged_df = transactions_df.merge(products_df, on='ProductID')
    plt.figure(figsize=(12, 5))
    merged_df.groupby('Category')['TotalValue'].sum().plot(kind='bar')
    plt.title('Sales by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../outputs/transaction_analysis.png')
    plt.close()

def generate_insights(customers_df, products_df, transactions_df):
    insights = [
        f"Total customers: {len(customers_df)}",
        f"Most common region: {customers_df['Region'].mode()[0]}",
        f"Average product price: ${products_df['Price'].mean():.2f}",
        f"Top product category: {products_df['Category'].mode()[0]}",
        f"Total revenue: ${transactions_df['TotalValue'].sum():.2f}",
        f"Average transaction value: ${transactions_df['TotalValue'].mean():.2f}"
    ]
    
    with open('../outputs/Akash_Ghosh_EDA.pdf', 'w') as f:
        f.write('\n\n'.join(insights))

def perform_eda(customers_df, products_df, transactions_df):
    os.makedirs('../outputs', exist_ok=True)
    analyze_customers(customers_df)
    analyze_products(products_df)
    analyze_transactions(transactions_df, products_df)
    generate_insights(customers_df, products_df, transactions_df)