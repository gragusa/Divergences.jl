# Divergences.el

[![Build Status](https://travis-ci.org/gragusa/Divergences.jl.svg?branch=master)](https://travis-ci.org/gragusa/Divergences.jl)

A Julia package for evaluating divergences.

It extends the ```Distances.jl``` package with Divergences that are useful in minimum divergence estimation techniques. 

The implemented divergences are:

* ```CressieRead```
* ```KullbackLeibler```
* ```ReverseKullbackLeibler```

## Examples

```evaluate(CressieRead(1), [1.1, 0.9, 0.8])```

```evaluate(CressieRead(1), [1.1, 0.9, 0.8], [0.8, .7, 1.1])```

