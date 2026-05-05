import display as d
from point2d import Point2D
from perceptron import Perceptron

# Hyperparameter, can tune this to try and get better results
MAX_ITER = 1000

if __name__ == "__main__":
    df = Point2D.csv_to_points("data/points.csv")
    p = Perceptron()
    for _ in range(MAX_ITER):
        p.train(df.to_numpy())
    print(p)
    d.display(p, df)
