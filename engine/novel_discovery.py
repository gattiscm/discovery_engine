import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN


def discover(

        memory_path=
        "results/novel_memory.csv",

        min_novel=20,

        eps=0.25,

        min_samples=5,

        normalize=True

):

    df = pd.read_csv(
        memory_path
    )

    if len(df) < min_novel:

        print(
            "\nNot enough novel samples."
        )

        return None

    latent_cols = [

        x

        for x in df.columns

        if x.startswith(
            "latent_"
        )

    ]

    X = df[
        latent_cols
    ].values

    # =====================================
    # LATENT NORMALIZATION
    # =====================================

    if normalize:
        X = (

                    X - np.mean(
                X,
                axis=0
            )

            ) / (

                    np.std(
                        X,
                        axis=0
                    )

                    + 1e-8

            )

    # =====================================
    # OPTIONAL NORMALIZATION
    # =====================================

    if normalize:

        X = (

            X - np.mean(
                X,
                axis=0
            )

        ) / (

            np.std(
                X,
                axis=0
            )

            + 1e-8

        )

    # =====================================
    # DBSCAN
    # =====================================

    cluster = DBSCAN(

        eps=eps,

        min_samples=min_samples

    )

    labels = cluster.fit_predict(
        X
    )

    noise_count = np.sum(
        labels == -1
    )

    clustered_count = np.sum(
        labels != -1
    )

    print()

    print(
        f"Noise samples: "
        f"{noise_count}"
    )

    print(
        f"Clustered samples: "
        f"{clustered_count}"
    )

    X = df[
        latent_cols
    ].values

    # normalize
    if normalize:
        X = (

                    X - np.mean(
                X,
                axis=0
            )

            ) / (

                    np.std(
                        X,
                        axis=0
                    )

                    + 1e-8

            )

    cluster = DBSCAN(

        eps=eps,

        min_samples=min_samples

    )

    labels = cluster.fit_predict(
        X
    )

    # diagnostics
    noise_count = np.sum(
        labels == -1
    )

    clustered_count = np.sum(
        labels != -1
    )

    print()

    print(
        f"Noise samples: "
        f"{noise_count}"
    )

    print(
        f"Clustered samples: "
        f"{clustered_count}"
    )

    df[
        "cluster"
    ] = labels

    print()

    print("=" * 60)
    print("DISCOVERED CANDIDATES")
    print("=" * 60)

    print()

    candidate_counts = {}

    for c in sorted(
            np.unique(labels)
    ):

        if c == -1:
            continue

        count = np.sum(
            labels == c
        )

        candidate_counts[
            int(c)
        ] = int(count)

        print(

            f"Candidate "
            f"{c}: "

            f"{count} "
            f"samples"

        )

    df.to_csv(

        "results/candidates.csv",

        index=False

    )

    return {

        "candidate_counts":
            candidate_counts,

        "labels":
            labels

    }