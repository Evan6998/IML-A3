import csv
import numpy as np
import argparse
import typing

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def review_to_vector(review: str, dictionary: dict[str, typing.Any]):
    x = review.split()
    x_trim = [word for word in x if word in dictionary]
    N = len(x_trim)
    return np.sum([dictionary[word] for word in x_trim], axis=0) / N


def embedding(data: str, glove_file):
    dataset = load_tsv_dataset(data)
    dictionary = load_feature_dictionary(glove_file)

    labels = np.array([row[0] for row in dataset], dtype=int).reshape(-1, 1)
    
    reviews = np.array([row[1] for row in dataset], dtype=str)
    feature_matrix = np.array([review_to_vector(review, dictionary) for review in reviews])

    print(f"{labels.shape=}")
    print(f"{feature_matrix.shape=}")
    return np.concatenate([labels.reshape(-1, 1), feature_matrix], axis=1)  # shape (8, 301)


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


def output_matrix(data, file):
    with open(file, 'w') as fout:
        for row in data:
            fout.write("\t".join([f"{round(float(v), 6):.6f}" for v in row]))
            fout.write("\n")


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()


    d = embedding(args.train_input, args.feature_dictionary_in)
    output_matrix(d, args.train_out)

    d = embedding(args.validation_input, args.feature_dictionary_in)
    output_matrix(d, args.validation_out)

    d = embedding(args.test_input, args.feature_dictionary_in)
    output_matrix(d, args.test_out)
