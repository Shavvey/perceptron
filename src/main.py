import display as d
from point2d import Point2D
from perceptron import Perceptron
from mlperceptron import Layer
from mlperceptron import MultiLayerPerceptron

# Hyperparameter, can tune this to try and get better results
MAX_ITER = 1000


def test_perceptron_train_and_display():
    # Get pandas dataframe of the 2d points
    df = Point2D.csv_to_points("data/points.csv")
    # Create perceptron
    p = Perceptron()
    for _ in range(MAX_ITER):
        # Train perceptron on 2D points based on {1,-1} features
        p.train(df.to_numpy())
    # Print final configuration to stdout
    print(p)


def test_multilayer_perceptron_iris():
    iris_data = Point2D.csv_to_points("data/iris_data.csv")
    # Return number of feature
    features = iris_data['species'].unique().tolist()
    num_features = len(features)
    print(iris_data[:-1])
    # Construct layers based on number of features
    hidden_layer = Layer([Perceptron(6) for _ in range(num_features)])
    # Create network with one single hidden layer
    network = MultiLayerPerceptron(hidden_layer)
    network.train(iris_data, features, 'species')


if __name__ == "__main__":
    test_multilayer_perceptron_iris()
