import pandas as pd
import os

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return False
    return True

def main():
    try:
        print("Starting data analysis...")
        
        data_path = './data'
        customers_file = os.path.join(data_path, 'Customers.csv')
        products_file = os.path.join(data_path, 'Products.csv')
        transactions_file = os.path.join(data_path, 'Transactions.csv')
        
        for file in [customers_file, products_file, transactions_file]:
            if not check_file_exists(file):
                return
        
        print("Loading data files...")
        customers_df = pd.read_csv(customers_file)
        print(f"Loaded Customers data: {len(customers_df)} records")
        
        products_df = pd.read_csv(products_file)
        print(f"Loaded Products data: {len(products_df)} records")
        
        transactions_df = pd.read_csv(transactions_file)
        print(f"Loaded Transactions data: {len(transactions_df)} records")
        
        if not os.path.exists('./outputs'):
            os.makedirs('./outputs')
            print("Created outputs directory")
        
        print("\nStarting EDA analysis...")
        from eda_analysis import perform_eda
        perform_eda(customers_df, products_df, transactions_df)
        print("EDA analysis completed")
        
        print("\nGenerating lookalike recommendations...")
        from lookalike_model import generate_lookalikes
        generate_lookalikes(customers_df, products_df, transactions_df)
        print("Lookalike recommendations completed")
        
        print("\nPerforming customer clustering...")
        from clustering_model import perform_clustering
        perform_clustering(customers_df, products_df, transactions_df)
        print("Clustering analysis completed")
        
        print("\nAll analyses completed successfully!")
        print("Check the 'outputs' directory for results")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()