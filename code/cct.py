import os
import pandas as pd
import pymc as pm


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

# Build the PyMC model
with pm.Model() as cct_model:
    # Prior for competence (D): one per informant
    D = pm.Uniform('D', lower=0.5, upper=1, shape=N)

    # Prior for consensus answers (Z): one per item
    Z = pm.Bernoulli('Z', p=0.5, shape=M)

    # Reshape D for broadcasting
    D_reshaped = D[:, None]  # shape: (N, 1)

    # Compute response probability matrix
    p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)  # shape: (N, M)

    # Likelihood
    X = pm.Bernoulli('X', p=p, observed=X_data)

    # Sample from the posterior
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9, return_inferencedata=True)
