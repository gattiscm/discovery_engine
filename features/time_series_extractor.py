import numpy as np


class TimeSeriesExtractor:

    def __init__(
            self,
            eps=1e-8
    ):
        self.eps = eps

    def extract(
            self,
            x
    ):
        x = np.asarray(
            x,
            dtype=float
        )

        dx = np.diff(
            x
        )

        fft = np.abs(np.fft.rfft(x))
        fft = fft / (np.sum(fft) + self.eps)


        dominant_freq_bin = int(
            np.argmax(
                fft
            )
        )

        energy = np.mean(x ** 2)

        peak_idx = int(
            np.argmax(
                np.abs(
                    x
                )
            )
        )

        mad = np.median(
            np.abs(
                x - np.median(x)
            )
        )

        iqr = (
                np.percentile(x, 75)
                -
                np.percentile(x, 25)
        )

        return {

            "mean":
                np.mean(x),

            "std":
                np.std(x),

            "min":
                np.min(x),

            "max":
                np.max(x),

            "mad":
                mad,

            "iqr":
                iqr,

            "amplitude":
                np.max(x) - np.min(x),

            "energy":
                energy,

            "rms":
                np.sqrt(
                    np.mean(
                        x**2
                    )
                ),

            "crest_factor":
                np.max(
                    np.abs(x)
                ) / (
                    np.sqrt(
                        np.mean(
                            x**2
                        )
                    )
                    +
                    self.eps
                ),

            "peak_abs":
                np.max(np.abs(x)),

            "peak_index":
                peak_idx / max(len(x) - 1, 1),

            "slope_mean":
                np.mean(dx),

            "slope_std":
                np.std(dx),

            "slope_max":
                np.max(dx),

            "slope_min":
                np.min(dx),

            "zero_crossings":
                np.sum(
                    np.diff(
                        np.signbit(x)
                    )
                ),

            "dominant_freq_bin":
                dominant_freq_bin,

            "spectral_energy":
                np.mean(
                    fft ** 2
                ),

            "spectral_entropy":
                self._spectral_entropy(
                    fft
                )
        }

    def _spectral_entropy(
            self,
            fft
    ):
        p = fft / (np.sum(fft) + self.eps)

        p = np.clip(
            p,
            self.eps,
            1.0
        )

        spectral_entropy = -np.sum(
            p * np.log(p)
        )

        spectral_entropy /= np.log(len(p))

        return spectral_entropy