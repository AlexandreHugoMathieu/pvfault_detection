# Project

Scripts to simulate PV defaults and to detect them.

The project is built for and includes the jupyter notebook used during DATASUN workshop 2022 on "PV fault detection"

## General Information

The repository is broadly inspired from pvlib and pvanalytics.

- pvlib: https://pvlib-python.readthedocs.io/en/stable/
- pvanalytics: https://pvanalytics.readthedocs.io/en/stable/

I sincerely thank those developers who offer the possibilities to other researchers to build from the top-edge methods.

This project:
- Simulates defaults: fixed shading, bypass diode short circuit, soiling and inverter clipping
- Includes detection methods to detect those defaults
- Implements and allows to fit a range of pv model such as the PAPM from King: King, D & Kratochvil, J & Boyson, W. (2004). Photovoltaic Array Performance Model. 8. 10.2172/919131.

## Setup

### Install Python 3.9

[python docs](https://docs.python.org/3/using/unix.html#getting-and-installing-the-latest-version-of-python)

### Virtual Environment

Instructions for setting up a virtual environment for your platform can be found in the [python docs](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

### PIP Dependencies

Once you have your virtual environment setup and running, install dependencies by running:

```bash
pip install -r requirements.txt
```

## Contact
Created by [@Alex](https://alexandrehugomathieu.github.io/alexandremathieu.github.io//) - feel free to contact me!
