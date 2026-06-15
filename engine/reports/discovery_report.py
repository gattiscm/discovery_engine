# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-09
# Updated: New.
# Purpose: Converts mixed-type tabular data into a numeric feature
#          matrix for DiscoveryModel, including numeric values,
#          z-score features, and one-hot categorical features.
# ######################################################################


from pathlib import Path

import json
import pandas as pd
import numpy as np


def json_safe(value):
    '''
        Safe conversion to JSON datatypes.

        @params  value: any

        @return value : any
    '''
    if isinstance(value,np.integer):
        return int(value)

    if isinstance(value,np.floating):
        return float(value)

    return value


class DiscoveryReport:
    # ######################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-09
    # Updated: New.
    # Purpose: Collection of methods to report on the discovery model
    #          results and methods.
    # ######################################################################

    def __init__(
            self,
            category,
            sample_count,
            comparison,
            closest_distribution,
            novelty_results):

        self.category = category
        self.sample_count = (sample_count)
        self.comparison = (comparison)
        self.closest_distribution = (closest_distribution)
        self.novelty_results = (novelty_results)
        self.increases = []
        self.decreases = []
        self._categorize()

    def __repr__(self):
        return (
            f"DiscoveryReport("
            f"category={self.category}, "
            f"samples={self.sample_count}, "
            f"novelty_rate={self.novelty_rate:.4f})"
        )

    def __str__(self):
        return self.__repr__()

    def _categorize(self):
        '''
            Categorizes data as increase or decrease based on text.

            @params na : na
        '''
        for row in self.comparison:
            arrow = row["arrow"]
            if arrow in ("⇈","↑"):
                self.increases.append(row)
            elif arrow in ("⇊","↓"):
                self.decreases.append(row)

    @property
    def novelty_rate(self):
        '''
            Purpose of function/method.

            @params na : na

            @return calculated : float
        '''
        if (len(self.novelty_results) == 0):
            return 0.0

        return sum(r["is_novel"] for r in self.novelty_results) / len(self.novelty_results)

    def increases_dataframe(self):
        '''
            Calculates and generates increased report data.

            @params na : na

            @return calculated : pandas dataframe
        '''
        if len(self.increases) == 0:
            return pd.DataFrame()

        rows = []
        for row in self.increases:
            contributors = ", ".join(c["feature"] for c in row[ "contributors"])
            rows.append({
                "Arrow": row["arrow"],
                "Latent": row["latent"],
                "Z Score":round(row["z_score"], 4),
                "Importance": round(row["importance"], 4),
                "Contributors": contributors
            })

        return pd.DataFrame(rows)

    def decreases_dataframe(self):
        '''
            Calculates and generates decreased report data.

            @params na : na

            @return cccc : pandas dataframe
        '''
        if len(self.decreases) == 0:
            return pd.DataFrame()
        rows = []

        for row in self.decreases:
            contributors = ", ".join(c["feature"] for c in row["contributors"])

            rows.append({
                "Arrow": row["arrow"],
                "Latent": row["latent"],
                "Z Score": round(row["z_score"], 4),
                "Importance": round(row["importance"], 4),
                "Contributors": contributors
            })

        return pd.DataFrame(rows)

    def summary(self):
        '''
            Prints data to screen based on results.
        '''
        print()
        print("=" * 80)
        print("DISCOVERY REPORT")
        print("=" * 80)
        print()
        print(f"Closest Category: {self.category}")
        print(f"Samples Evaluated: {self.sample_count}")
        print(f"Novelty Rate: {self.novelty_rate:.2%}")
        print()
        print("Closest Category Distribution")
        print("-" * 80)
        print(self.closest_distribution)
        print()
        print("Strongest Increases")
        print("-" * 80)

        if len(self.increases) > 0:
            print(self.increases_dataframe().to_string(index=False))

        else:
            print("None")

        print()
        print("Strongest Decreases")
        print("-" * 80)

        if len(self.decreases) > 0:
            print(self.decreases_dataframe().to_string(index=False))

        else:
            print("None")

        print()
        print("=" * 80)
        print()


    def to_json(self,filepath=None):
        '''
            Converts data to JSON and saves JSON.

            @params filepath : KWARG, string

            @return payload : JSON data
        '''
        payload = {
            "category": json_safe(self.category),
            "sample_count": json_safe(self.sample_count),
            "novelty_rate": json_safe(self.novelty_rate),
            "closest_distribution": json_safe(self.closest_distribution.to_dict()),
            "strongest_increases": json_safe(self.increases),
            "strongest_decreases": json_safe(self.decreases)
        }

        if filepath is None:
            return payload
        filepath = (Path(filepath))
        with open(filepath, "w",encoding="utf-8") as f:
            json.dump(payload,f,indent=4,ensure_ascii=False)

    def to_csv(self,filepath):
        '''
            Saves data to a CSV.

            @params filepath : KWARG, string
        '''

        rows = []

        for row in self.increases:
            rows.append({
                "Group": "Increase",
                "Arrow": row["arrow"],
                "Latent": row["latent"],
                "Z Score": row["z_score"],
                "Importance": row["importance"],
                "Contributors": "; ".join(c["feature"] for c in row["contributors"])
            })

        for row in self.decreases:
            rows.append({
                "Group": "Decrease",
                "Arrow": row["arrow"],
                "Latent": row["latent"],
                "Z Score": row["z_score"],
                "Importance": row["importance"],
                "Contributors": "; ".join(c["feature"] for c in row["contributors"])
            })

        pd.DataFrame(rows).to_csv(filepath,index=False)