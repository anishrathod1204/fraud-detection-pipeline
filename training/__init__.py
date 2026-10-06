"""Model training for the fraud-detection pipeline.

Everything under this package runs offline, in batch, to produce the artifact
bundle that the streaming scorer loads: a fitted feature scaler, an
IsolationForest, an autoencoder, and the recall-first decision threshold chosen
on held-out data. The package deliberately depends only on NumPy and pandas so
that training is reproducible in this environment without a network install.
"""
