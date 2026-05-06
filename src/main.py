import display as d
from point2d import Point2D
from perceptron import Perceptron
from mlperceptron import Layer
from mlperceptron import MultiLayerPerceptron
import stats as stats

# Hyperparameter, can tune this to try and get better results
MAX_ITER = 500


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
    classes = iris_data["species"].unique().tolist()
    num_features = len(iris_data.columns) - 1
    num_classes = len(classes)
    # Construct neurons in layers based on number of features in data set
    hidden_layer = Layer([Perceptron(num_features) for _ in range(num_classes)])
    # Create network with single hidden layer
    network = MultiLayerPerceptron(hidden_layer)
    for _ in range(MAX_ITER):
        network.train(iris_data, classes, "species")
    d = {0: classes[0], 1: classes[1], 2: classes[2]}
    preds = network.test(iris_data, d)
    print(
        f"Total percent accuracy of network: {stats.percent_acc(preds, iris_data['species'].tolist()) * 100:.02f}%"
    )
    stats.display_confusion_matrix(preds, iris_data['species'].tolist(), classes)


if __name__ == "__main__":
    test_multilayer_perceptron_iris()
