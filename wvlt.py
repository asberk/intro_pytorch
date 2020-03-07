import pywt as wt


class WaveletTransformer:
    def __init__(self, wname, dim=1):
        self.wname = wname
        self.dim = dim
        # Determine decomposition, reconstruction and output_format definitions
        if self.dim == 1:
            self._wavedec = wt.wavedec
            self._waverec = wt.waverec
            self._of = "wavedec"
        elif self.dim == 2:
            self._wavedec = wt.wavedec2
            self._waverec = wt.waverec2
            self._of = "wavedec2"
        else:
            self._wavedec = wt.wavedecn
            self._waverec = wt.waverecn
            self._of = "wavedecn"

    def decompose(self, y):
        """
        WaveletTransformer.decompose(self, y)

        Computes and returns the 1d ravelled wavelet transform of y using the
        wavelet specified by self.wname.

        Inputs
        ------
        y : self.dim numpy array
            A signal given a numpy array with ndim = self.dim
        wname : str

        Returns
        -------
        coef_ravelled : 1d numpy array
        slices : list
        shapes : list

        Example
        -------
        >>> # not tested:
        >>> x0 = np.random.randn(128)
        >>> wt = WaveletTransformer('db1', x0.ndim)
        >>> coefs, *aux_data = wt.decompose(x0)
        >>> print(coefs.ndim)
        """
        assert (
            y.ndim == self.dim
        ), f"Expected y of dimension {self.dim} but got {y.ndim}."
        return wt.ravel_coeffs(self._wavedec(y, self.wname))

    def reconstruct(self, coefs, *args):
        """
        WaveletTransformer.reconstruct(self, coef, *args)

        Computes and returns the inverse wavelet transform of the ravelled
        vector of coefficients coefs using the wavelet specified by self.wname.

        Inputs
        ------
        coef : 1d numpy array
        args: tuple
            Expecting, e.g., (slices, shapes) as returned by pywt.ravel_coeffs

        Returns
        -------
        y : numpy array

        Example
        -------
        >>> # not tested:
        >>> xHat = wt.reconstruct(coefs, *aux_data)
        >>> print(nnse(xHat, x0, 1))
        """
        return self._waverec(
            wt.unravel_coeffs(coefs, *args, output_format=self._of), self.wname
        )


def verify_transforms(decompose, reconstruct):
    assert (decompose is None) or callable(decompose), (
        f"Expected callable or None for decompose but got "
        f"{type(decompose)}."
    )
    assert (reconstruct is None) or callable(reconstruct), (
        f"Expected callable or None for reconstruct but got "
        f"{type(reconstruct)}."
    )
    assert not (callable(decompose) ^ callable(reconstruct)), (
        f"Exactly one of decompose and reconstruct is not callable, "
        f"but both or neither must be."
    )
    return
