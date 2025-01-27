import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

def analyze_customers(customers_df, output_path):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=customers_df, x='Region')
    plt.title('Customer Distribution by Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'customer_distribution.png'))
    plt.close()
    
    return {
        'total_customers': len(customers_df),
        'regions': customers_df['Region'].nunique(),
        'top_region': customers_df['Region'].mode()[0]
    }

def analyze_products(products_df, output_path):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=products_df, x='Category', y='Price')
    plt.title('Price Distribution by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'price_distribution.png'))
    plt.close()
    
    return {
        'total_products': len(products_df),
        'categories': products_df['Category'].nunique(),
        'avg_price': products_df['Price'].mean(),
        'price_range': products_df['Price'].max() - products_df['Price'].min()
    }

def analyze_transactions(transactions_df, output_path):
    plt.figure(figsize=(10, 6))
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    monthly_sales = transactions_df.groupby(transactions_df['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()
    monthly_sales.plot(kind='line')
    plt.title('Monthly Sales Trend')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'sales_trend.png'))
    plt.close()
    
    return {
        'total_transactions': len(transactions_df),
        'total_revenue': transactions_df['TotalValue'].sum(),
        'avg_transaction': transactions_df['TotalValue'].mean(),
        'total_quantity': transactions_df['Quantity'].sum()
    }

def generate_pdf_report(customer_stats, product_stats, transaction_stats, output_path):
    pdf_path = os.path.join(output_path, 'Akash_Ghosh_EDA.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("E-Commerce Data Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    
    story.append(Paragraph("Customer Analysis", styles['Heading2']))
    story.append(Paragraph(f"Total Customers: {customer_stats['total_customers']}", styles['Normal']))
    story.append(Paragraph(f"Number of Regions: {customer_stats['regions']}", styles['Normal']))
    story.append(Paragraph(f"Most Popular Region: {customer_stats['top_region']}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    
    story.append(Paragraph("Product Analysis", styles['Heading2']))
    story.append(Paragraph(f"Total Products: {product_stats['total_products']}", styles['Normal']))
    story.append(Paragraph(f"Number of Categories: {product_stats['categories']}", styles['Normal']))
    story.append(Paragraph(f"Average Price: ${product_stats['avg_price']:.2f}", styles['Normal']))
    story.append(Paragraph(f"Price Range: ${product_stats['price_range']:.2f}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    
    story.append(Paragraph("Transaction Analysis", styles['Heading2']))
    story.append(Paragraph(f"Total Transactions: {transaction_stats['total_transactions']}", styles['Normal']))
    story.append(Paragraph(f"Total Revenue: ${transaction_stats['total_revenue']:,.2f}", styles['Normal']))
    story.append(Paragraph(f"Average Transaction Value: ${transaction_stats['avg_transaction']:.2f}", styles['Normal']))
    story.append(Paragraph(f"Total Quantity Sold: {transaction_stats['total_quantity']}", styles['Normal']))
    
    doc.build(story)

def perform_eda(customers_df, products_df, transactions_df):
    output_path = os.environ.get('PROJECT_OUTPUT_PATH', 'outputs')
    os.makedirs(output_path, exist_ok=True)
    
    customer_stats = analyze_customers(customers_df, output_path)
    product_stats = analyze_products(products_df, output_path)
    transaction_stats = analyze_transactions(transactions_df, output_path)
    
    generate_pdf_report(customer_stats, product_stats, transaction_stats, output_path)