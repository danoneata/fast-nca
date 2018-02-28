import argparse
import pdb

from functools import partial

from matplotlib import pyplot as plt

import numpy as np

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

from sklearn.datasets import (
    load_digits,
    load_iris,
    load_wine,
)

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import (
    Bunch,
    check_random_state,
    shuffle as util_shuffle,
)

from nca import NCA


SEED = 1337
TEST_SIZE = 0.3
N_NEIGHBORS = 1


class Euclidean(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X


def make_circles(
        n_samples=500, n_classes=5, shuffle=True, noise=None,
        random_state=None, factor_step=0.2):

    M = n_samples // n_classes
    generator = check_random_state(random_state)
    linspace = np.linspace(0, 2 * np.pi, M + 1)[:-1]

    def _get_X(i):
        factor = 1 + i * factor_step
        circ_x = np.cos(linspace) * factor
        circ_y = np.sin(linspace) * factor
        return np.vstack((circ_x, circ_y, np.ones(M))).T

    def _get_y(i):
        return i * np.ones(M, dtype=np.intp)

    X = np.vstack(_get_X(i) for i in range(n_classes))
    y = np.hstack(_get_y(i) for i in range(n_classes))

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        eps = generator.normal(scale=noise, size=X.shape)
        eps[:, 2] *= 10 * n_classes  # Extra noise on the third dimension
        X += eps

    return Bunch(data=X, target=y)


DATA_LOADERS = {
    'wine': load_wine,
    'iris': load_iris,
    'circles': partial(make_circles, n_samples=5000, noise=0.05, factor_step=0.5),
}


MODELS = {
    'nca': NCA(dim=None),
    'nca-2d': NCA(dim=2),
    'euclidean': Euclidean(),
    'pca': PCA(),
    'pca-2d': PCA(n_components=2),
}


def main():

    parser = argparse.ArgumentParser(
        description='Apply the kNN classifier using different metrics.',
    )

    parser.add_argument(
        '-m', '--model',
        choices=MODELS,
        default='nca',
        help='what to do',
    )
    parser.add_argument(
        '-d', '--data',
        choices=DATA_LOADERS,
        default='wine',
        help='on which data to run the model',
    )
    parser.add_argument(
        '--to-plot',
        action='store_true',
        help='plot the projected data',
    )
    parser.add_argument(
        '--seed',
        default=SEED,
        type=int,
        help='seed to fix the randomness',
    )
    parser.add_argument(
        '-v', '--verbose',
        default=0,
        action='count',
        help='how much information to output',
    )

    args = parser.parse_args()
    np.random.seed(args.seed)
    
    data = DATA_LOADERS[args.data]()
    X, y = data.data, data.target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=args.seed)

    # Apply metric model
    model = MODELS[args.model]
    X_tr = model.fit_transform(X_tr, y_tr)
    X_te = model.transform(X_te)

    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn.fit(X_tr, y_tr)
    y_pr = knn.predict(X_te)

    accuracy = 100 * accuracy_score(y_te, y_pr)
    print('Test accuracy: {:.2f}%'.format(accuracy))

    if args.to_plot:
        plt.scatter(X_te[:, 0], X_te[:, 1], c=y_te, s=40)
        plt.show()


if __name__ == '__main__':
    main()
