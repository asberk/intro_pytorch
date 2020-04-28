# intro_pytorch

I've taken some pains to write a bunch of boilerplate that will make working
with models and data much easier. This has the unfortunate consequence of making
the code look more complicated when you're new to looking at it. I hope I've
struck a decent balance between simplifying the initial introduction and
simplifying the workflow later on when you realize you need more/better tools.

See `small_network.py` for a simple example. See `wavelet_example.py` for an
intermediate example.


## Requirements

* I'm using Python 3.6.10. Other versions not tested but probably fine.

* [PyTorch](https://pytorch.org/): On a Mac this simply looks like `pip install
  torch torchvision` (with no `cuda` installation). 
  * Note: at time of writing, `torch` does not cooperate with PIL version
    7.0.0. To remedy this, you can run `pip install 'PIL<7.0.0`
* [PyWavelets](https://pypi.org/project/PyWavelets/): `pip install PyWavelets`
* [sklearn](https://scikit-learn.org/stable/): `pip install -U scikit-learn`
* `numpy`, `pandas`, `matplotlib`
* `torchsummary`: `pip install torchsummary`


## Get this repo running


Fork the [intro_pytorch repo](https://github.com/asberk/intro_pytorch). When
you're in the forked repo on your own GitHub profile, click the green "Clone or
download" button and then click the copy-to-clipboard icon. Next, open up a
terminal window, navigate to your favourite directory and run:

```bash
git clone `pbpaste`
cd intro_pytorch/
jupyter notebook
```

If you're not using a Mac, then substitute ``pbpaste`` for the link that was
copied to your clipboard.


