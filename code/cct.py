import os
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np

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
    D_reshaped = D[:, None]

    # Compute response probability matrix
    p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)

    # Likelihood
    X = pm.Bernoulli('X', p=p, observed=X_data)

    # Sample from the posterior
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9, return_inferencedata=True)

# Summarize results
summary = az.summary(trace, var_names=["D", "Z"])
print(summary)

# Plot posterior distributions
az.plot_posterior(trace, var_names=["D"])
az.plot_posterior(trace, var_names=["Z"])

# Posterior means for consensus answers (Di) and competence (Zj)
posterior_D = trace.posterior['D'].mean(dim=["chain", "draw"]).values
posterior_Z = trace.posterior['Z'].mean(dim=["chain", "draw"]).values
consensus_estimate = np.round(posterior_Z).astype(int)

# Compare with naive majority vote
majority_vote = (X_data.mean(axis=0) > 0.5).astype(int)

print("Naive Majority Vote:", majority_vote)
print("CCT Consensus Estimate:", consensus_estimate)

# Identify most and least competent informants
most_competent = np.argmax(posterior_D)
least_competent = np.argmin(posterior_D)
print(f"Most competent informant: {most_competent}, D = {posterior_D[most_competent]:.3f}")
print(f"Least competent informant: {least_competent}, D = {posterior_D[least_competent]:.3f}")