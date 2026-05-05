import pandas as pd


class Point2D:
    x: float = 0
    y: float = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def csv_to_points(csv_filename: str) -> pd.DataFrame:
        df = pd.read_csv(csv_filename)
        return df
