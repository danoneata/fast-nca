# Fast NCA

Fast implementation of the [Neighborhood Component Analysis](https://papers.nips.cc/paper/2566-neighbourhood-components-analysis.pdf) algorithm in Python.

# Examples

sklearn-like API:

```python
from nca import NCA
nca_model = NCA()
nca_model.fit(X, y)
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
- [ ] Add examples
- [ ] Add example on MNIST
- [ ] Add some visualizations
- [ ] Add tests
- [ ] Package the code
- [ ] Compute timings
- [ ] Big O notation for memory and computation
- [ ] Write documentation
- [ ] Implement version with mini-batches
- [ ] Provide links to other implementations and outline differences
- [ ] Motivate metric learning
- [ ] Test numerical stability
