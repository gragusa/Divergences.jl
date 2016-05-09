# Divergences.el

[![Build Status](https://travis-ci.org/gragusa/Divergences.jl.svg?branch=master)](https://travis-ci.org/gragusa/Divergences.jl)

`Divergences` is a Julia package that makes it easy to evaluate the value of divergences and their derivatives. These divergences are used to good effects in the package [MomentBasedEstimators](http://github.com/gragusa/MomentBasedEstimators.jl/git). 

The package defines a `Divergence` type with the following suptypes:

* ```CressieRead```
* ```KullbackLeibler```
* ```ReverseKullbackLeibler```

