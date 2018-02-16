import pdb

import numpy as np

from sklearn.datasets import (
    load_digits,
    load_iris,
)

from sklearn.metrics import accuracy_score

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.neighbors import KNeighborsClassifier

from nca import NCA


SEED = 42


def evaluate(X_tr, X_te, y_tr, y_te, learn_nca):

    if learn_nca:
        nca = NCA(reg=0.1)
        X_tr = nca.fit_transform(X_tr, y_tr)
        X_te = nca.transform(X_te)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_tr, y_tr)
    y_pr = knn.predict(X_te)
    return 100 * accuracy_score(y_te, y_pr)


def main():
    
    data = load_digits()
    X, y = data.data, data.target

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    for learn_nca in (True, False):
        accuracy = evaluate(X_tr, X_te, y_tr, y_te, learn_nca)
        print('Using NCA {}. Test accuracy: {:.2f}'.format(learn_nca, accuracy))


if __name__ == '__main__':
    main()
