from pathlib import Path

import json
import numpy as np
import pandas as pd


class LatentRelationshipGraph:

    def __init__(
            self,
            latent_vectors
    ):

        self.Z = np.asarray(
            latent_vectors,
            dtype=float
        )

        self.n_latents = (
            self.Z.shape[1]
        )

        self.relationship_matrix = (
            self._build_matrix()
        )

    def _build_matrix(
            self
    ):

        return np.corrcoef(
            self.Z,
            rowvar=False
        )

    def strongest_relationships(
            self,
            top_k=20,
            min_strength=0.50
    ):

        results = []

        for i in range(
                self.n_latents
        ):

            for j in range(
                    i + 1,
                    self.n_latents
            ):

                score = (
                    self.relationship_matrix[
                        i,
                        j
                    ]
                )

                if np.isnan(
                        score
                ):
                    continue

                if (
                        abs(score)
                        <
                        min_strength
                ):
                    continue

                results.append({

                    "latent_a":
                        int(i),

                    "latent_b":
                        int(j),

                    "correlation":
                        float(score),

                    "strength":
                        abs(score)

                })

        results.sort(
            key=lambda x:
            x["strength"],
            reverse=True
        )

        return results[
               :top_k
               ]

    def to_dataframe(
            self
    ):

        labels = [

            f"Latent_{i}"

            for i in range(
                self.n_latents
            )

        ]

        return pd.DataFrame(
            self.relationship_matrix,
            index=labels,
            columns=labels
        )

    def to_csv(
            self,
            filepath
    ):

        self.to_dataframe(
        ).to_csv(
            filepath
        )

    def to_json(
            self,
            filepath=None
    ):

        payload = {

            "relationships":
                self.strongest_relationships()

        }

        if filepath is None:

            return payload

        filepath = Path(
            filepath
        )

        with open(
                filepath,
                "w",
                encoding="utf-8"
        ) as f:

            json.dump(
                payload,
                f,
                indent=4
            )

    def summary(
            self,
            top_k=20
    ):

        print()
        print(
            "Latent Relationships"
        )

        print(
            "=" * 80
        )

        for row in self.strongest_relationships(
                top_k=top_k
        ):

            corr = row[
                "correlation"
            ]

            if corr > 0:

                relation = "↔"

            else:

                relation = "⇎"

            print(

                f"Latent_{row['latent_a']} "
                f"{relation} "
                f"Latent_{row['latent_b']} "
                f"({corr:.4f})"

            )

        print(
            "=" * 80
        )