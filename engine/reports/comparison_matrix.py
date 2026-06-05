from pathlib import Path

import json
import pandas as pd


class ComparisonMatrix:

    def __init__(
            self,
            comparison_results
    ):

        self.results = (
            comparison_results
        )

        self.labels = [

            f"Latent_{row['latent']}"

            for row in comparison_results

        ]

        self.matrix = (
            self._build_matrix()
        )

    def _build_matrix(
            self
    ):

        matrix = []

        for i, row_i in enumerate(
                self.results
        ):

            current_row = []

            for j, row_j in enumerate(
                    self.results
            ):

                if i == j:

                    current_row.append(
                        "---"
                    )

                    continue

                zi = row_i[
                    "z_score"
                ]

                zj = row_j[
                    "z_score"
                ]

                if (
                        zi > 0
                        and
                        zj > 0
                ):

                    current_row.append(
                        "⇈"
                    )

                elif (
                        zi < 0
                        and
                        zj < 0
                ):

                    current_row.append(
                        "⇊"
                    )

                elif (
                        abs(zi) < 0.5
                        and
                        abs(zj) < 0.5
                ):

                    current_row.append(
                        "↔"
                    )

                else:

                    current_row.append(
                        "⇎"
                    )

            matrix.append(
                current_row
            )

        return matrix

    def to_dataframe(
            self
    ):

        return pd.DataFrame(
            self.matrix,
            index=self.labels,
            columns=self.labels
        )

    def to_text(
            self
    ):

        return (
            self.to_dataframe()
            .to_string()
        )

    def to_csv(
            self,
            filepath
    ):

        filepath = Path(
            filepath
        )

        self.to_dataframe(
        ).to_csv(
            filepath
        )

    def to_json(
            self,
            filepath=None
    ):

        payload = {

            "latents":
                self.labels,

            "matrix":
                self.matrix

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
                indent=4,
                ensure_ascii=False
            )

    def summary(
            self
    ):

        print()
        print(
            "Comparison Matrix"
        )

        print(
            "=" * 80
        )

        print(
            self.to_text()
        )

        print()
        print(
            "=" * 80
        )

        print()
        print(
            "Latent Contributors"
        )

        print(
            "=" * 80
        )

        for row in self.results:

            print()

            print(
                f"Latent_{row['latent']} "
                f"{row['arrow']} "
                f"(z={row['z_score']:.2f})"
            )

            for contributor in row[
                "contributors"
            ]:

                print(

                    f"    ↦ "
                    f"{contributor['feature']} "
                    f"("
                    f"{contributor['correlation']:.3f}"
                    f")"

                )

        print()