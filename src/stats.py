from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def percent_acc(preds: list[str], actuals: list[str]) -> float:
    incorrect = 0
    correct = 0
    for i, pred in enumerate(preds):
        if pred == actuals[i]:
            correct += 1
        else:
            incorrect += 1
    return (correct) / (incorrect + correct)


def display_confusion_matrix(preds: list[str], actuals: list[str], classes: list[str]):
    cm = confusion_matrix(actuals, preds, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot()
    plt.show()
