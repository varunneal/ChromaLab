# ColorMatching

`models.py` contains this workflow:

- acquire spectral absorption curves for pigments (e.g. cmyk)
- find a target color in some basis (e.g. (0.3, 0.6, 0.9) in RGB)
- use backprop to find concentrations of pigments closest to RGB

Checkout `spectra.py` for color mixing details. `main.py` has some plots. 