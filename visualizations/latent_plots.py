import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA


IMAGE_DIR = "images"

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)


def plot_latent_space(
        Z,
        labels,
        filename,

        title="Latent Space",

        alpha=.8
):

    pca = PCA(
        n_components=2
    )

    Z2 = pca.fit_transform(
        Z
    )

    plt.figure(
        figsize=(8,8)
    )

    unique_labels = sorted(
        np.unique(labels)
    )

    for label in unique_labels:

        mask = (
            np.array(labels)
            ==
            label
        )

        plt.scatter(

            Z2[mask,0],
            Z2[mask,1],

            label=str(label),

            alpha=alpha
        )

    plt.title(
        title
    )

    plt.xlabel(
        "PCA Component 1"
    )

    plt.ylabel(
        "PCA Component 2"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

def plot_known_vs_hidden(
        Z_known,
        y_known,

        Z_hidden,

        filename
):

    pca = PCA(
        n_components=2
    )

    Z_all = np.vstack([
        Z_known,
        Z_hidden
    ])

    Z2 = pca.fit_transform(
        Z_all
    )

    n_known = len(
        Z_known
    )

    known_2d = Z2[:n_known]

    hidden_2d = Z2[n_known:]

    plt.figure(
        figsize=(10,10)
    )

    unique_labels = sorted(
        np.unique(y_known)
    )

    for label in unique_labels:

        mask = (
            np.array(y_known)
            ==
            label
        )

        plt.scatter(

            known_2d[mask,0],
            known_2d[mask,1],

            label=f"Known {label}",

            alpha=.7
        )

    plt.scatter(

        hidden_2d[:,0],
        hidden_2d[:,1],

        marker="x",

        s=120,

        label="Hidden",

        alpha=.9
    )

    plt.title(
        "Known vs Hidden Latent Space"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    def plot_novelty_histogram(
            known_scores,
            hidden_scores,

            filename,

            title="Novelty Distribution"
    ):

        plt.figure(
            figsize=(8, 6)
        )

        plt.hist(

            known_scores,

            bins=30,

            alpha=.7,

            label="Known"

        )

        plt.hist(

            hidden_scores,

            bins=30,

            alpha=.7,

            label="Hidden"

        )

        plt.title(
            title
        )

        plt.xlabel(
            "Novelty Score"
        )

        plt.ylabel(
            "Count"
        )

        plt.legend()

        plt.tight_layout()

        path = os.path.join(
            IMAGE_DIR,
            filename
        )

        plt.savefig(
            path,
            dpi=300
        )

        plt.close()

        print(
            f"Saved: {path}"
        )

    def plot_hidden_candidates(

            Z_hidden,

            candidate_labels,

            filename,

            title=
            "Hidden Candidate Latent Space"

    ):

        pca = PCA(
            n_components=2
        )

        Z2 = pca.fit_transform(
            Z_hidden
        )

        plt.figure(
            figsize=(8, 8)
        )

        labels = np.asarray(
            candidate_labels
        )

        for label in sorted(
                np.unique(labels)
        ):
            mask = (
                    labels
                    ==
                    label
            )

            plt.scatter(

                Z2[mask, 0],
                Z2[mask, 1],

                label=
                f"Candidate {label}",

                alpha=.8
            )

        plt.title(
            title
        )

        plt.xlabel(
            "PCA Component 1"
        )

        plt.ylabel(
            "PCA Component 2"
        )

        plt.legend()

        plt.tight_layout()

        path = os.path.join(
            IMAGE_DIR,
            filename
        )

        plt.savefig(
            path,
            dpi=300
        )

        plt.close()

        print(
            f"Saved: {path}"
        )

def plot_novelty_histogram(
        known_scores,
        hidden_scores,

        filename,

        title="Novelty Distribution"
):

    plt.figure(
        figsize=(8,6)
    )

    plt.hist(

        known_scores,

        bins=30,

        alpha=.7,

        label="Known"

    )

    plt.hist(

        hidden_scores,

        bins=30,

        alpha=.7,

        label="Hidden"

    )

    plt.title(
        title
    )

    plt.xlabel(
        "Novelty Score"
    )

    plt.ylabel(
        "Count"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


def plot_hidden_candidates(

        Z_hidden,

        candidate_labels,

        filename,

        title=
            "Hidden Candidate Latent Space"

):

    pca = PCA(
        n_components=2
    )

    Z2 = pca.fit_transform(
        Z_hidden
    )

    plt.figure(
        figsize=(8,8)
    )

    labels = np.asarray(
        candidate_labels
    )

    for label in sorted(
            np.unique(labels)
    ):

        mask = (
            labels
            ==
            label
        )

        plt.scatter(

            Z2[mask,0],
            Z2[mask,1],

            label=
                f"Candidate {label}",

            alpha=.8
        )

    plt.title(
        title
    )

    plt.xlabel(
        "PCA Component 1"
    )

    plt.ylabel(
        "PCA Component 2"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )