# Fast NCA

Fast implementation of the [Neighborhood Component Analysis](https://papers.nips.cc/paper/2566-neighbourhood-components-analysis.pdf) algorithm in Python.
The ideas behind some of the choices are further expanded in Master's thesis, [Fast low-rank metric learning](http://homepages.inf.ed.ac.uk/imurray2/projects/2011_dan_oneata_msc.pdf).

Features:

- Sklearn-like API
- Same gradient cost as the objective function
- Avoid overflows when the scale of the metric is large
- WIP Mini-batch version

# Examples

Sample usage from Python:

```python
from nca import NCA
n = NCA()
n.fit(X, y)
X = n.transform(X)
```

For an example, run the [example.py](example.py) script.
Among others the script accepts the type of model and the dataset:

```bash
python example.py --model nca --data wine
```

For a complete description of the available options, just invoke the help prompt:

```bash
python example.py -h
```

# Installation

The code depends on the usual Python scientific environment: NumPy, SciPy, Scikit-learn.
The required packages are listed in the `requirements.txt` file and can be installed in a virtual environment as follows:

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

Special thanks to [Iain Murray](http://homepages.inf.ed.ac.uk/imurray2/) for writing a first version of this code, teaching me about [automatic differentiation](http://www.inf.ed.ac.uk/teaching/courses/mlpr/2017/notes/w5a_backprop.html) and supervising my Master's thesis project.

# TODO

- [x] Add requirements
- [x] Add examples
- [x] Add example using NCA with Nearest Neighbour
- [x] Test numerical stability
- [x] Add argument parsing for example script
- [x] Add some visualizations
- [x] Add PCA to the list of models
- [x] Add concentric circles and noise to the list of datasets
- [ ] Create separate modules, for example: data, models
- [ ] Package the code
- [ ] Do not compute gradients when only the cost function is required
- [ ] Add example on MNIST
- [ ] Add tests
- [ ] Add gradient check tests
- [ ] Compute timings
- [ ] Big O notation for memory and computation
- [ ] Write documentation
- [ ] Implement version with mini-batches
- [ ] Provide links to other implementations and outline differences
- [ ] Motivate metric learning
- [ ] Implement [nearest mean metric learning](https://hal.inria.fr/hal-00817211/document)
