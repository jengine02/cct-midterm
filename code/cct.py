import pandas as pd
import os

# Load the data
def load_data():
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'plant_knowledge.csv')
    df = pd.read_csv(filepath)
    print("Columns in the CSV:", df.columns)  # Debugging line

    data = df.drop(columns=['Informant']).values  # shape: (N, M)
    return data

# Load data
X_data = load_data()
N, M = X_data.shape
