# exlipbab

Code for the paper "ExLipBaB: Exact Lipschitz Constant Computation for  Piecewise Linear Neural Networks"


## Installation

The package was developed under python 3.11.14 and can be installed with:

```bash
$ pip install .
```

we recommend first creating a virtual environment, for example through
```bash
$ conda create -n exlipbab python=3.11
```

Note: On some systems, we encountered errors when trying to install pycddlib automatically as above. In such cases, try manually installing pyddlib (e.g. version 3.0.2) through

```bash
$ python -m pip install pycddlib

```

before installing exlipbab.



## Usage

The code for the examples of the paper can be found in the applications Folder and should mostly run as-is with an installed exlipbab package. They are named <model_name>_exlipbab_computation.py.
Some data or code fragments, we cannot redistribute, so we link to the corresponding repositories.

The code for the comparative results with LipSDP can be found in the LipSDP_applications Folder, but needs the corresponding repositories to function correctly, which are described in the corresponding matlab files.

## Credits

The package structure of `exlipbab` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
