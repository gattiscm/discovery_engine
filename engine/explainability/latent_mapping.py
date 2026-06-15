# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-14
# Updated: New.
# Purpose: Maps latent dimensions to their strongest
#          correlated input features.
# ######################################################################

import numpy as np


def build_latent_map( X,Z,top_features=5):
    '''
        Builds feature-to-latent correlation map.

        @params X : numpy array
            Original input feature matrix.

                Z : numpy array
            Latent representation matrix.

                top_features : int
            Number of strongest feature
            contributors retained.

        @return mapping : dict
            Mapping of latent dimensions to
            correlated input features.
    '''
    X = np.asarray(X,dtype=float)
    Z = np.asarray(Z,dtype=float)
    n_inputs = X.shape[1]
    n_latents = Z.shape[1]
    correlations = np.zeros((n_inputs,n_latents))

    for i in range(n_inputs):
        for j in range(n_latents):
            c = np.corrcoef(X[:, i],Z[:, j])[0, 1]
            if np.isnan(c):
                c = 0.0
            correlations[i,j] = c

    mapping = {}

    for latent_idx in range(n_latents):
        ranked = np.argsort(np.abs(correlations[:,latent_idx]))[::-1]
        contributors = []

        for feature_idx in ranked[:top_features]:
            contributors.append({
                "feature_index": int(feature_idx),
                "correlation": float(correlations[feature_idx,latent_idx])
            })

        mapping[latent_idx] = contributors

    return mapping