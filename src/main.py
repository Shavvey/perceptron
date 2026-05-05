from point2d import Point2D
import pandas as pd

if __name__ == "__main__":
    df = Point2D.csv_to_points("data/points.csv")
    print(df)
