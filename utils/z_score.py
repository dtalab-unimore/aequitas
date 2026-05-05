import numpy as np
import pandas as pd


class RobustZScoreNormalization:
    def __init__(self, df, eps=1e-12):
        self.eps = eps
        self.z_score_cols = df.columns[2:-1]  # Exclude 'date', 'instrument' and 'Label' columns
        self.median = df.median(numeric_only=True)
        abs_dev = df[self.z_score_cols].transform(lambda x: (x - x.median()).abs())
        self.mad = abs_dev.median(numeric_only=True) + self.eps

    def robust_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        mad = self.mad * 1.4826  # Scale MAD to match standard deviation
        df[self.z_score_cols] = (df[self.z_score_cols] - self.median[self.z_score_cols]) / mad[self.z_score_cols]
        df[self.z_score_cols] = np.clip(df[self.z_score_cols], -3, 3)
        return df

    def cszscore(self, x):
        return (x - x.mean()).div(x.std())