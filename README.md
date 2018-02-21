# Fast NCA

Fast implementation of the [Neighborhood Component Analysis](https://papers.nips.cc/paper/2566-neighbourhood-components-analysis.pdf) algorithm in Python.

# Examples

sklearn-like API:

Sample usage:

```python
from nca import NCA
n = NCA()
n.fit(X, y)
X = n.transform(X)
```

Run the [example.py](example.py) file:

```bash
python example.py
```

# Installation

- NumPy
- SciPy
- Scikit-learn

If you want to use a virtual environment, just run the following commands:

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

# Metric learning

# Benchmarks

# Related work

- Other NCA implementations Python + MATLAB
- Other metric learning algorithms

# Acknowledgements

Iain Murray for supervision and writing a first version of this code.

# TODO

- [x] Add requirements
- [x] Add examples
- [x] Add example using NCA with Nearest Neighbour
- [x] Test numerical stability
- [x] Add argument parsing for example script
- [x] Add some visualizations
- [ ] Do not compute gradients when only the cost function is required
- [ ] Add example on MNIST
- [ ] Add tests
- [ ] Add gradient check tests
- [ ] Package the code
- [ ] Compute timings
- [ ] Big O notation for memory and computation
- [ ] Write documentation
- [ ] Implement version with mini-batches
- [ ] Provide links to other implementations and outline differences
- [ ] Motivate metric learning
- [ ] Implement [nearest mean metric learning](https://hal.inria.fr/hal-00817211/document)
