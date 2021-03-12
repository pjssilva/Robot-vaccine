# Robot Vaccine

## Warning

The code is is derived from the Robot Dance project tayored to analyze vaccine deployment.

You can use it to reproduce the experiments in the companion manuscript.

## Installing dependencies

This code is written in [Julia](https://www.julialang.org) and
[Python](https://www.python.org), so you need both installed. To install Julia you should
go to its website, select download and follow instructions. After installing Julia you need
to use its package manager and install the `JuMP` and `Ipopt` packages. 

To get good performance out of Ipopt, it is very important to use a high performance linear
solver. The default free solver, `mumps` works well for 10 cities or so. If you want to
solve larger problems with tens of cities you will need to compile Ipopt with support to
[HSL](http://www.hsl.rl.ac.uk) linear solvers. Robot dance uses MA97 in the code whenever
available. I have also prepared a little [document](compiling_ipopt.md) describing how to
compile Ipopt with HSL under Ubuntu.

There are many ways to install Python. I use
[Anaconda](https://www.anaconda.com/products/individual). By installing Anaconda you get a
comprehensive and updated Python environment with packages the code use like `numpy`,
`scipy` , and `matplotlib`.

After that install [PyJulia](https://github.com/JuliaPy/pyjulia).

## Running and input files

Different experiments are made using different notebooks. If you need assistence, get in touch.

### Computational resources.

The code uses a highly parallel optimization solver to run a large scale optimization
problem, named Ipopt, which is installed as a Julia package. This demands a good computer.
A problem with more than 10-20 cities/regions the solver may face difficulties and stall.
In order to avoid this you may need to install Ipopt with HSL. HSL is free for research.
Please look at these [instructions of how to compile Ipopt](compiling_ipopt.md) if needed.
Also long time horizons (like the 400 days simulations we did in he report) are very
demanding. The code is not ready to run on a cluster. We will try to continuously improve
the code in order to overcome these limitations.

## Copyright 

Copyright Paulo J. S. Silva, Luis Gustavo Nonato, and Marcelo Córdova. See the [license file](LICENSE.md).

## Funding

This research is supported by CeMEAI/FAPESP, Instituto Serrapilheira, and CNPq.

## Please Cite Us

We provide this code hoping that it will be useful for others. Please if you use it, let us
know about you experiences. Moreover, if you use this code in any publication, please cite
us. This is very important. For the moment we only have the manuscript, so cite it as

Nonato, L. G; Peixoto, P.; Pereira, T.; Sagastizabal, C.; Silva, P. J. S. "Robot
Dance: a mathematical optimization platform for intervention against Covid-19 in
a complex network". [Optimization Online, 2020.](http://www.optimization-online.org/DB_HTML/2020/10/8054.html) 
