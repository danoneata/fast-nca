import numpy as np
from nca import NCA


def main():
    
    data = np.loadtxt('wine.txt')
    yy = data[:, -1]
    xx = data[:, :-1]

    nca = NCA()
    nca.fit(xx, yy)
    nca.transform(xx)


if __name__ == '__main__':
    main()
