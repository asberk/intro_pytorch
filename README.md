# intro_pytorch

I've taken some pains to write a bunch of boilerplate that will make working
with models and data much easier. This has the unfortunate consequence of making
the code look more complicated when you're new to looking at it. I hope I've
struck a decent balance between simplifying the initial introduction and
simplifying the workflow later on when you realize you need more/better tools.

See `small_network.py` for a simple example. See `wavelet_example.py` for an
intermediate example.


## Requirements

* I'm using Python 3.6.10. Other versions not tested

* [PyTorch](https://pytorch.org/): On a Mac this simply looks like `pip install
  torch torchvision` (with no `cuda` installation). 
  * Note: at time of writing, `torch` does not cooperate with PIL version
    7.0.0. To remedy this, you can run `pip install 'PIL<7.0.0`
* [PyWavelets](https://pypi.org/project/PyWavelets/): `pip install PyWavelets`
* [sklearn](https://scikit-learn.org/stable/): `pip install -U scikit-learn`
* `numpy`, `pandas`, `matplotlib`
