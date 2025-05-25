import numpy as np
import argparse

THRESHOLD = 0.5

def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    for _ in range(num_epoch):
        for ith in range(X.shape[0]):
            gradient = (sigmoid(X[ith] @ theta) - y[ith]) * X[ith]
            theta -= learning_rate * gradient


def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    return sigmoid(X @ theta) > THRESHOLD


def compute_error(
    y_pred : np.ndarray,
    y : np.ndarray
) -> float:
    return float(np.mean(y != y_pred))


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    train_data = np.genfromtxt(args.train_input, delimiter="\t")
    val_data = np.genfromtxt(args.validation_input, delimiter="\t")
    test_data = np.genfromtxt(args.test_input, delimiter="\t")

    # training: 
    labels, features = train_data[:, 0], train_data[:, 1:]
    N, D = features.shape
    features = np.concatenate([features, np.ones(((N, 1)))], axis=1)
    theta = np.zeros(D+1)
    train(theta, features, labels, args.num_epoch, args.learning_rate)  

    # predict training dataset
    predict_labels = predict(theta, features)
    training_error = compute_error(predict_labels, labels) 
    with open(args.train_out, 'w') as fout:
        for label in predict_labels:
            fout.write(f"{int(label)}\n")
    
    # predict testing dataset
    labels, features = test_data[:, 0], test_data[:, 1:]
    N, D = features.shape
    features = np.concatenate([features, np.ones(((N, 1)))], axis=1)
    predict_labels = predict(theta, features)
    test_error = compute_error(predict_labels, labels)
    with open(args.test_out, 'w') as fout:
        for label in predict_labels:
            fout.write(f"{int(label)}\n")

    with open(args.metrics_out, 'w') as fout:
        fout.write(f"error(train): {training_error:.6f}\n")
        fout.write(f"error(test): {test_error:.6f}")
