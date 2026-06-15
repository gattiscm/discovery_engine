# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-15
# Updated: Explainability, category memory, latent mapping,
#          novelty thresholding, and reporting support added.
# Purpose: Primary Discovery Engine model responsible for
#          feature processing, latent representation learning,
#          novelty detection, explainability, category
#          comparison, and automated discovery reporting.
# ######################################################################


import numpy as np
import pandas as pd

from features.feature_space import FeatureSpace
from engine.explainability.latent_mapping import build_latent_map
from engine.models.category_memory import CategoryMemory
from engine.explainability.novelty import compute_novelty
from engine.explainability.purity import compute_purity
from engine.explainability.attribution import compute_feature_contributions
from engine.explainability.attribution import compute_group_contributions
from engine.reports.discovery_report import (DiscoveryReport)
from engine.explainability.latent_relationships import (LatentRelationshipGraph)


from engine.explainability.arrows import z_to_arrow


class DiscoveryModel:
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-15
    # Updated: Expanded explainability and reporting support.
    # Purpose: Core Discovery Engine model that learns latent
    #          representations, performs category matching,
    #          novelty detection, attribution analysis,
    #          relationship discovery, and report generation.
    # ##################################################################

    def __init__(
            self,
            novelty_mode="percentile",
            novelty_threshold=95,
            eps=1e-8,
            activation=None,
            novelty=None,
            optimizer=None,
            use_encoder=False,
            encoder_type="tabular",
            encoder_kwargs=None,
            minor_cutoff=0.5,
            major_cutoff=2.0
    ):
        '''
            Initializes DiscoveryModel.

            @params novelty_mode : str
                    Threshold calculation mode.

                    novelty_threshold : float
                    Threshold value or percentile.

                    eps : float
                    Numerical stability constant.

                    activation : object
                    Optional activation function.

                    novelty : object
                    Optional novelty metric.

                    optimizer : object
                    Optional optimizer.

                    use_encoder : bool
                    Enables feature extraction encoder.

                    encoder_type : str
                    Encoder implementation type.

                    encoder_kwargs : dict
                    Encoder configuration parameters.

            @return None
        '''
        self.layers = []
        self.novelty_cutoff_ = None
        self.training_novelty_scores_ = None
        self.groups = {}
        self.mode = "parallel"
        self.merge = "concat"
        self.novel_memory = None
        self.feature_space = FeatureSpace()
        self.memory = CategoryMemory()
        self.activation = activation
        self.novelty = novelty
        self.optimizer = optimizer
        self.use_encoder = use_encoder
        self.encoder_type = encoder_type
        self.encoder_kwargs = encoder_kwargs or {}
        self.is_fit = False
        self.minor_cutoff = minor_cutoff
        self.major_cutoff = major_cutoff
        self.feature_names = None
        self.latent_map = None
        self.extractor = None
        self.novelty_mode = novelty_mode
        self.novelty_threshold = novelty_threshold

        if major_cutoff <= minor_cutoff:
            raise ValueError("major_cutoff must be greater than minor_cutoff.")

    def _score_latent(self, z):
        scores = {}

        for category in self.memory.categories():
            reconstruction = (self.memory.reconstruct(category))

            if self.novelty is not None:
                novelty = self.novelty.compute(z,reconstruction)

            else:
                novelty = compute_novelty(z,reconstruction)

            scores[category] = novelty

        best = min(scores,key=scores.get)

        return {
            "closest": best,
            "novelty": scores[best],
            "scores": scores
        }

    def _learn_novelty_cutoff(self, Z):

        training_scores = []

        for z in Z:
            result = self._score_latent(z)
            training_scores.append(result["novelty"])

        self.training_novelty_scores_ = (np.asarray(training_scores))
        self.novelty_cutoff_ = (self.compute_threshold(self.training_novelty_scores_))

        print()
        print(f"Stored novelty cutoff: {self.novelty_cutoff_:.4f}")
        print()

    def _require_cutoff(self):

        if self.novelty_cutoff_ is None:
            raise RuntimeError("Novelty cutoff not learned. Call fit() first.")

    def _build_encoder(self):

        if self.extractor is not None:
            return

        if not self.use_encoder:
            return

        if self.encoder_type == "tabular":

            from features.tabular_extractor import (TabularExtractor)

            self.extractor = TabularExtractor(**self.encoder_kwargs)

        elif self.encoder_type == "timeseries":           #### Placeholder for time-series expansion
            '''
            from features.time_series_extractor import (
                TimeSeriesExtractor
            )

            self.extractor = TimeSeriesExtractor(
                **self.encoder_kwargs
            )'''
            raise NotImplementedError('This is a placeholder for future development: time-series data.')

        elif self.encoder_type == "shape":                #### Placeholder for shapes expansion
            '''
            from features.shape_extractor import (
                ShapeExtractor
            )

            self.extractor = ShapeExtractor(
                **self.encoder_kwargs
            )'''
            raise NotImplementedError('This is a placeholder for future development: shape data.')

        else:
            raise ValueError( f"Unknown encoder: {self.encoder_type}")

    def add(self, layer):
        self.layers.append(layer)

    def add_neuron_group(self, name, columns):
        self.groups[name] = columns

    def summary(self):
        '''
            Displays model configuration, learned categories,
            novelty settings, and training status.

            @params None

            @return None
        '''
        print("DiscoveryModel Summary")
        print("=" * 60)

        print(f"Fit status: {self.is_fit}")
        print(f"Mode: {self.mode}")
        print(f"Merge: {self.merge}")
        print(f"Layers: {len(self.layers)}")

        for i, layer in enumerate(self.layers):
            print(f"  Layer {i}: {layer.__class__.__name__}")

        if self.feature_names is not None:
            print(f"Input features: {len(self.feature_names)}")
        else:
            print("Input features: unknown")

        if self.memory is not None and self.is_fit:
            categories = list(self.memory.categories())
            print(f"Categories: {len(categories)}")
            print(f"Category names: {categories}")

        print(f"Novelty mode: {self.novelty_mode}")
        print(f"Novelty threshold: {self.novelty_threshold}")

        if self.novelty_cutoff_ is not None:
            print(f"Stored novelty cutoff: {self.novelty_cutoff_:.4f}")
        else:
            print("Stored novelty cutoff: not learned")

        print(f"Latent map: {self.latent_map is not None}")

        if hasattr(self, "novel_memory") and self.novel_memory is not None:
            if hasattr(self.novel_memory, "samples"):
                print(f"Novel memory samples: {len(self.novel_memory.samples)}")

        print("=" * 60)

    def relationship_graph(
            self,
            X
    ):
        '''
            Builds a latent relationship graph from samples.

            @params X : numpy array
                    Feature matrix.

            @return graph : LatentRelationshipGraph
        '''
        Z = self.latent(X)

        return (LatentRelationshipGraph(Z))

    def compute_threshold(self,novelty_scores):
        novelty_scores = np.asarray(novelty_scores)

        if self.novelty_mode == "fixed":
            return self.novelty_threshold

        elif self.novelty_mode == "std":
            return (np.mean(novelty_scores) + np.std(novelty_scores) )

        elif self.novelty_mode == "percentile":
            return np.percentile(novelty_scores,self.novelty_threshold)

        else:
            raise ValueError("Unknown novelty mode.")

    def build(self, mode="parallel", merge="concat"):
        self.mode = mode
        self.merge = merge

        for layer in self.layers:
            if hasattr(layer, "groups") and layer.groups is None:
                layer.groups = self.groups if self.groups else None

            if hasattr(layer, "mode"):
                layer.mode = mode

            if hasattr(layer, "merge"):
                layer.merge = merge

        return self

    def set_feature_names(self,names):

        self.feature_names = list(names)

        return self

    def _forward_layers(self, X):
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def set_extractor(self,extractor):
        self.extractor = extractor

    def _extract_features(self,X):

        if self.extractor is None:
            return X

        X = np.asarray(X,dtype=float)

        if (self.extractor is not None and hasattr(self.extractor,"get_feature_names")):
            self.feature_names = (self.extractor.get_feature_names())

        rows = []

        for x in X:
            features = (self.extractor.extract(x))
            rows.append(list(features.values()))

        return np.array(rows)

    def fit(self,X,y,epochs=10):
        '''
            Fits the DiscoveryModel to training data.

            Learns latent representations, category memory,
            latent mappings, and novelty thresholds.

            @params X : numpy array
                    Feature matrix.

                    y : numpy array
                    Category labels.

                    epochs : int
                    Training epochs.

            @return self : DiscoveryModel
        '''

        # BUILD ENCODER IF REQUESTED

        self._build_encoder()

        # FIT EXTRACTOR

        if self.extractor is not None:
            try:
                self.extractor.fit(X,feature_names=self.feature_names)
            except TypeError:
                self.extractor.fit(X)

        X = np.asarray(X,dtype=float)
        X_scaled = (self.feature_space.fit_transform(X))
        X_processed = (self._extract_features(X_scaled))

        print(f"Processed shape: {X_processed.shape}")

        if self.novelty is not None:
            Z_init = self._forward_layers(X_processed)

            self.novelty.fit(Z_init)

        for epoch in range(epochs):

            Z = (self._forward_layers(X_processed))
            self.latent_map = (build_latent_map(X_processed,Z))
            classes = np.unique(y)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_encoded = np.array([class_to_idx[c] for c in y])
            target = np.eye(len(classes))[y_encoded]

            loss = np.mean((Z[:,:target.shape[1]] - target) ** 2)
            grad_small = (2 * (Z[:,:target.shape[1]] - target) / len(X_processed))
            grad = np.zeros_like(Z)
            grad[:,:target.shape[1]] = grad_small

            for layer in reversed(self.layers):
                grad = layer.backward(grad)

            if epoch % 2 == 0:
                print(f"Epoch {epoch} Loss: {loss:.4f}")

        Z = (self._forward_layers(X_processed))
        self.memory.fit(Z,y)
        self._learn_novelty_cutoff(Z)
        self.is_fit = True

        return self

    def predict(self,X):
        '''
            Predicts the closest known category for samples.

            @params X : numpy array
                    Feature matrix.

            @return predictions : numpy array
        '''

        X = np.asarray(X,dtype=float)
        X_scaled = (self.feature_space.transform(X))
        X_processed = (self._extract_features(X_scaled))

        Z = (self._forward_layers(X_processed))
        predictions = []

        for z in Z:
            scores = {}
            for category in (self.memory.categories()):
                reconstruction = (self.memory.reconstruct(category))

                if self.novelty is not None:
                    novelty = self.novelty.compute(z,reconstruction)
                else:
                    novelty = compute_novelty(z,reconstruction)

                scores[category] = novelty

            best = min(scores,key=scores.get)
            predictions.append(best)

        return np.array(predictions)


    def latent(
            self,
            X
    ):
        '''
            Returns latent-space representations for samples.

            @params X : numpy array
                    Feature matrix.

            @return Z : numpy array
                    Latent representations.
        '''
        X_scaled = (self.feature_space.transform(X))
        X_processed = (self._extract_features(X_scaled))

        return self._forward_layers(X_processed)

    def latent_contributors(self,latent_idx):

        if (self.latent_map is None or latent_idx not in self.latent_map):
            return []

        contributors = []

        for row in self.latent_map[latent_idx]:
            feature_idx = row[ "feature_index"]

            if (self.feature_names and feature_idx < len(self.feature_names)):
                name = (self.feature_names[feature_idx])
            else:
                name = (f"Feature_{feature_idx}")

            contributors.append({
                "feature": name,
                "feature_index": feature_idx,
                "correlation": row["correlation"]
            })

        return contributors

    def detect_novelty(self,X):
        '''
            Detects novel samples using the learned novelty
            threshold.

            @params X : numpy array
                    Feature matrix.

            @return results : list
                    Novelty analysis results.
        '''
        self._require_cutoff()
        X = np.asarray(X,dtype=float)
        X_scaled = (self.feature_space.transform(X))

        X_processed = (self._extract_features( X_scaled))
        Z = (self._forward_layers(X_processed))
        threshold = (self.novelty_cutoff_)

        print()
        print(f"Using stored cutoff: {threshold:.4f}")
        print()

        results = []

        for z in Z:
            result = self._score_latent(z)
            is_novel = (result["novelty"] > threshold)

            if (is_novel and self.novel_memory is not None):
                self.novel_memory.store(
                    latent=z,
                    novelty=result["novelty"],
                    closest=result["closest"])

            results.append({
                "closest":result[ "closest"],
                "novelty": result["novelty"],
                "is_novel": is_novel
            })

        return results

    def compare_to_category(self,X,category,top_k=10,eps=1e-8):
        '''
            Compares samples against a known category and
            identifies the most influential latent differences.

            @params X : numpy array
                    Feature matrix.

                    category : object
                    Target category.

                    top_k : int
                    Number of latent dimensions returned.

                    eps : float
                    Numerical stability constant.

            @return results : list
        '''
        if not self.is_fit:
            raise RuntimeError("Model must be fit first.")

        if category not in self.memory.categories():
            raise ValueError(f"Unknown category: {category}")

        X = np.asarray(X,dtype=float)
        Z = self.latent(X)
        reconstruction = (self.memory.reconstruct(category))
        delta = (Z - reconstruction)
        mean_delta = np.mean(delta,axis=0)
        std_delta = np.std(delta,axis=0)
        z_scores = (mean_delta / (std_delta + eps))
        ranked = np.argsort(np.abs(z_scores))[::-1]
        results = []

        for latent_idx in ranked[:top_k]:
            z_score = z_scores[latent_idx]
            raw_delta = mean_delta[latent_idx]
            contributors = (self.latent_contributors(latent_idx))

            results.append({
                "latent": int(latent_idx),
                "contributors": contributors,
                "delta": float(raw_delta),
                "z_score": float(z_score),
                "arrow": z_to_arrow(z_score,minor_cutoff=self.minor_cutoff,major_cutoff=self.major_cutoff),
                "importance": abs(z_score)
            })

        return results


    def auto_discovery_report(self,X,top_k=20):
        '''
            Generates an automated discovery report for a
            collection of samples.

            Combines novelty detection, category comparison,
            and attribution analysis into a structured report.

            @params X : numpy array
                    Feature matrix.

                    top_k : int
                    Number of latent dimensions reported.

            @return report : DiscoveryReport
        '''
        results = self.detect_novelty(X)
        closest_distribution = (pd.Series([r["closest"] for r in results]).value_counts())
        closest = (closest_distribution.idxmax())
        comparison = (self.compare_to_category(X,closest,top_k=top_k))

        return DiscoveryReport(
            category=closest,
            sample_count=len(X),
            comparison=comparison,
            closest_distribution=closest_distribution,
            novelty_results=results
        )

    def explain(self, x):
        '''
            Produces an explainability report for a sample.

            Calculates novelty, purity, feature attribution,
            residual analysis, and group contributions for
            every known category.

            @params x : numpy array
                    Single sample.

            @return report : dict
        '''
        if not self.is_fit:
            raise ValueError("DiscoveryModel must be fit before explain().")

        x = np.asarray(x,dtype=float).reshape(1, -1)
        x_scaled = ( self.feature_space.transform(x))
        x_processed = (self._extract_features(x_scaled))

        # use transformed feature count

        n_features = x_processed.shape[1]
        z = (self._forward_layers(x_processed)[0])
        report = {}

        for category in self.memory.categories():
            reconstruction = self.memory.reconstruct(category)

            if self.novelty is not None:
                novelty = self.novelty.compute(z,reconstruction)

            else:
                novelty = compute_novelty(z,reconstruction)

            purity = compute_purity(novelty)
            contrib = compute_feature_contributions(z,reconstruction)
            latent_contrib = contrib["magnitude"]
            latent_direction = contrib["direction"]
            latent_signed = contrib["signed"]
            input_contrib = np.zeros(n_features)
            input_direction = np.zeros(n_features)

            # GROUPED INPUT MODE

            if len(self.groups) > 0:
                total_latent = np.sum(latent_contrib)
                if total_latent == 0:
                    total_latent = 1.0

                group_names = list(self.groups.keys())

                for group_idx, group_name in enumerate(group_names):
                    indices = self.groups[group_name]
                    valid = [i for i in indices if i < n_features]
                    if len(valid) == 0:
                        continue

                    # Map each input group to a slice of latent space when possible.

                    start = int(group_idx * len(latent_contrib) / len(group_names))
                    end = int((group_idx + 1) * len(latent_contrib) / len(group_names))
                    latent_slice = latent_contrib[start:end]
                    direction_slice = latent_direction[start:end]

                    if len(latent_slice) == 0:
                        group_strength = (total_latent / len(group_names))
                        group_direction = 0

                    else:
                        group_strength = np.sum(latent_slice)
                        group_direction = np.sign(np.mean(direction_slice))

                    per_input = (group_strength / len(valid))
                    input_contrib[valid] = per_input
                    input_direction[valid] = group_direction

            # NON-GROUPED INPUT MODE

            else:
                n = min(len(latent_contrib),n_features)
                input_contrib[:n] = latent_contrib[:n]
                input_direction[:n] = latent_direction[:n]

            # NORMALIZE INPUT CONTRIBUTIONS

            total = np.sum(input_contrib)
            if total > 0:
                input_contrib = (input_contrib / total)

            # GROUP CONTRIBUTIONS

            if len(self.groups) > 0:
                group_contrib = compute_group_contributions(input_contrib,groups=self.groups)
            else:
                group_contrib = {}

            report[category] = {
                "novelty": novelty,
                "purity": purity,
                "feature_contributions": input_contrib.tolist(),
                "feature_direction": input_direction.tolist(),
                "signed_residual": latent_signed.tolist(),
                "group_contributions": group_contrib
            }

        best = min(report,key=lambda c: report[c]["novelty"])

        return {
            "prediction": best,
            "categories": report
        }

    def pretty_explain(self,x):
        '''
            Prints a human-readable explainability report.

            @params x : numpy array
                    Single sample.

            @return None
        '''
        report = self.explain(x)
        print()
        print(f"Prediction: {report['prediction']}")
        print()

        for category, data in (report["categories"].items()):
            print(f"Category {category}")
            print(f"Novelty: {data['novelty']:.4f}")
            print(f"Purity: {data['purity']:.4f}")

            # Input contributions

            print("\nInput Contributions:")
            contribs = (data["feature_contributions"])
            directions = (data["feature_direction"])

            for i in range(len(contribs)):
                mag = contribs[i]
                dir_ = 0
                if i < len(directions):
                    dir_ = directions[i]
                if dir_ > 0:
                    relation = ("positive shift")
                elif dir_ < 0:
                    relation = ("negative shift")
                else:
                    relation = ("neutral")

                # use external names if available

                name = (f"Feature_{i}")
                if (self.feature_names and i < len(self.feature_names)):
                    name = (self.feature_names[i])
                print(f"  {name}: {mag:.3f} ({relation})")

            # Top drivers

            pairs = list(enumerate(contribs))
            pairs.sort(key=lambda x: x[1],reverse=True)
            print("\nTop Drivers:")

            for i, val in (pairs[:3]):
                name = (f"Feature_{i}")
                if (self.feature_names and i < len(self.feature_names)):
                    name = (self.feature_names[i])
                print(f"  {name} ({val:.3f})")

            # Group contributions

            if len(data["group_contributions"]) > 0:
                print("\nGroup Contributions:")
                for name, value in (data["group_contributions"].items()):
                    print(f"  {name}: {value:.3f}")

            print()