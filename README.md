Install via:

```
pip install git+https://github.com/VarunNSrivastava/ChromaLab
```
**Basics**

This library is based around the `Spectra` object. 
`Spectra` are easy to plot, convert to RGB, XYZ, and combine in various ways.

To initialize a `Spectra`, pass in numpy arrays corresponding to wavelengths and data:

```
from chromalab.observer import Spectra
import numpy as np

wavelengths = np.arange(400, 701, 1)
data = np.zeros_like(wavelengths)
data[190:] = 1

red = Spectra(wavelengths=wavelengths, data=data)
print(red.to_rgb())
```

If you want to pass in data outside 0 and 1 make sure to set `normalized` to `false`.

This library implements `Cone`, `Pigment`, and `Illuminant` as extensions of `Spectra`,
each equipped with some helpful methods. 

**Cone Fundamentals:**

Stockman and Sharpe (2000):

```
from chromalab.observer import Cone, Observer
import matplotlib.pyplot as plt
import numpy as np

wavelengths = np.arange(400, 701, 1)  # set any wavelengths you want

Cone.s_cone(wavelengths).plot()
Cone.m_cone(wavelengths).plot()
Cone.l_cone(wavelengths).plot()

plt.show()

```

From nomogram:

```
from chromalab.observer import Cone, Observer
import matplotlib.pyplot as plt
import numpy as np

wavelengths = np.arange(400, 701, 1)  # set any wavelengths you want

for template in ["neitz", "govardovskii"]:
    Cone.m_cone(wavelengths, template=template).plot()

plt.show()

```

Construct a cone fundamental from scratch:

```
from chromalab.observer import Cone, Observer
from chromalab.nomograms import GovardovskiiNomogram 
import matplotlib.pyplot as plt
import numpy as np

GovardovskiiNomogram(wavelengths1,530).with_preceptoral(od=0.35, lens=1, macular=0).plot(name="No macular pigment")
plt.legend()
plt.show()
```

The Cone class has a variety of helpful instance methods. It is an extension of the Spectra class. 
Spectra are easy to plot, convert to RGB, XYZ, and combine in various ways.

**Ink mixing:**

- `InkGamut` models the full gamut of a set of ink primaries
- you can either pass in ink primaries as a list of primaries (where each ink is a Spectra) or as a `Neugebauer` object
- If you pass in the former, it will generate a `Neugebauer` object using kubelka-munk interpolation
- If you've measured all the Neugebauer primaries, you can initialize your own `Neugebauer` object using a (key, Spectra) dictionary with keys corresponding to the binary codes of the primaries
- You should specify an illuminant as well, or it will default to equal-energy

Example usage:

```
from chromalab.spectra import Spectra, Illuminant
from chromalab.observer import Observer
from chromalab.inks import InkGamut
import numpy as np

cmy = # ... load neugebauer primaries dict. For cmy keys will be length 3 binary sequences

wavelengths1 = np.arange(390, 701, 1)
d65 = Illuminant.get("D65")
trichromat = Observer.trichromat()

cmy_neug = CellNeugebauer(cmy)
cmy_gamut = InkGamut(cmy, illuminant=d65)
point_cloud, percentages = cmy_gamut.get_point_cloud(trichromat)

```


**Plotting**

Plotting is the most helpful built-in function of `Spectra` objects and its descendents, `Cone`, `Pigment`, and `Illuminant`.

(hint: this feature works great in jupyter notebooks)

You can pass in no parameters and it will automatically color the plot according to its RGB code. 

```
from chromalab.observer import Spectra
import matplotlib.pyplot as plt
import numpy as np

wavelengths = np.arange(400, 701, 1)
data = np.zeros_like(wavelengths)
data[190:] = 1

red = Spectra(wavelengths=wavelengths, data=data)
red.plot()
plt.show()
```

Additionally, you can specify parameters like `name`, `color`, `alpha`, and `axis`. To use `name` remember to call
`plt.legend()`. Using a legend will automatically overwrite RGB coloring. 

```
from chromalab.spectra import Spectra
from chromalab.observer import Cone
import matplotlib.pyplot as plt
import numpy as np

wavelengths = np.arange(400, 701, 1)
data = np.zeros_like(wavelengths)
data[190:] = 1

red = Spectra(wavelengths=wavelengths, data=data)
cyan = 1 - red
red.plot(name="red")
cyan.plot(name="cyan")

Cone.m_cone().plot(name="m cone")

plt.legend()
plt.show()

```