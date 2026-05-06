def percent_acc(preds: list[int]) -> float:
    incorrect = 0
    correct = 0
    for pred in preds:
        if pred == 1:
            correct += 1
        else:
            incorrect += 1
    return correct / (correct + incorrect)
