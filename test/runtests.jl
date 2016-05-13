using Divergences
using Base.Test

# write your own tests here
@test 1 == 1

cr = CressieRead(2)

srand(1)
v = rand(10)
v = 10*v./sum(v)

x = rand(10)
x = 10*x./sum(x)

evaluate(cr, v)
evaluate(cr, v, x)

gradient(cr, 1.1)
gradient(cr, 0.9)

hessian(cr, 1.0)
hessian(cr, 1.0)



################################################################################
##
## Function - Cressie Read
##
################################################################################


seq =0:.1:3
cr = CressieRead(2.)

ch =[0.333333, 0.2835, 0.234667, 0.187833, 0.144, 0.104167, 0.0693333, 0.0405,
	 0.0186667, 0.00483333, 0., 0.00516667, 0.0213333, 0.0495, 0.0906667,
	 0.145833, 0.216, 0.302167, 0.405333, 0.5265, 0.666667, 0.826833, 1.008,
	 1.21117, 1.43733, 1.6875, 1.96267, 2.26383, 2.592, 2.94817, 3.33333]

[@test_approx_eq_eps ch[index] evaluate(cr, [value]) 1.e-4
									for (index, value) in enumerate(seq)];


cr = CressieRead(.5)

ch = [0.666667, 0.50883, 0.385924, 0.285756, 0.203976, 0.138071, 0.086344, 0.0475494,
	 0.0207223, 0.00508662, 0., 0.00491964, 0.0193789, 0.0429707, 0.0753365, 0.116156,
	 0.165144, 0.222038, 0.286605, 0.358626, 0.437903, 0.524252, 0.617503, 0.717497,
	 0.824085, 0.937129, 1.0565, 1.18207, 1.31373, 1.45136, 1.59487];

[@test_approx_eq_eps ch[index] evaluate(cr, [value]) 1.e-4
									for (index, value) in enumerate(seq)];

cr = CressieRead(-.5)

ch = [2., 0.935089, 0.611146, 0.40911, 0.270178, 0.171573, 0.101613, 0.0533599,
	  0.0222912, 0.00526681, 0., 0.00476461, 0.0182195, 0.0392983, 0.0671362,
	  0.101021, 0.140356, 0.184638, 0.233437, 0.28638, 0.343146, 0.403449,
	  0.467041, 0.5337, 0.603227, 0.675445, 0.750194, 0.827329, 0.90672, 0.988245, 1.0718]

[@test_approx_eq_eps ch[index] evaluate(cr, [value]) 1.e-4
									for (index, value) in enumerate(seq)];


################################################################################
##
## Gradient - Cressie Read
##
################################################################################

cr = CressieRead(2.)

ch = [-0.5, -0.495, -0.48, -0.455, -0.42, -0.375, -0.32, -0.255, -0.18,
	  -0.095, 0., 0.105, 0.22, 0.345, 0.48, 0.625, 0.78, 0.945, 1.12,
	  1.305, 1.5, 1.705, 1.92, 2.145, 2.38, 2.625, 2.88, 3.145, 3.42, 3.705, 4.]

[@test_approx_eq_eps ch[index] gradient(cr, value) 1.e-4
									for (index, value) in enumerate(seq)];

cr = CressieRead(.5)

ch = [-2., -1.36754, -1.10557, -0.904555, -0.735089, -0.585786, -0.450807,
-0.32668, -0.211146, -0.102633, 0., 0.0976177, 0.19089, 0.280351, 0.366432,
0.44949, 0.529822, 0.607681, 0.683282, 0.75681, 0.828427, 0.898275,
0.966479, 1.03315, 1.09839, 1.16228, 1.2249, 1.28634,  1.34664, 1.40588, 1.4641]

[@test_approx_eq_eps ch[index] gradient(cr, value) 1.e-4
									for (index, value) in enumerate(seq)];

cr = CressieRead(-.5)

ch = [-Inf, -4.32456, -2.47214, -1.65148, -1.16228, -0.828427, -0.581989, -0.390457,
	  -0.236068, -0.108185, 0., 0.0930748, 0.174258, 0.245884, 0.309691,
	  0.367007, 0.418861, 0.46607, 0.509288, 0.549047, 0.585786, 0.619869,
	  0.6516, 0.681239, 0.709006, 0.735089, 0.759653, 0.782839, 0.804771,
	  0.82556, 0.84529]

[@test_approx_eq_eps ch[index] gradient(cr, value) 1.e-4
									for (index, value) in enumerate(seq)];


###############################################################################
##
## Hessian - Cressie Read
##
#############################################################################
cr = CressieRead(2.)
ch = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4,
   1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.]
[@test_approx_eq_eps ch[index] hessian(cr, value) 1.e-4
									for (index, value) in enumerate(seq)];

cr = CressieRead(-.5)
ch = [+Inf, 31.6228, 11.1803, 6.08581, 3.95285, 2.82843, 2.15166, 1.70747,
      1.39754, 1.17121, 1., 0.866784, 0.760726, 0.67466, 0.603682, 0.544331,
      0.494106, 0.451156, 0.414087, 0.38183, 0.353553, 0.328603, 0.306454,
      0.286687, 0.268957, 0.252982, 0.238528, 0.2254, 0.213434, 0.20249, 0.19245]
[@test_approx_eq_eps ch[index] hessian(cr, value) 1.e-4
									for (index, value) in enumerate(seq)];



###############################################################################
##
## Function - Modified Cressie Read
##
#############################################################################

cr = ModifiedCressieRead(2, .2)

ch = [0.333333, 0.2835, 0.234667, 0.187833, 0.144, 0.104167, 0.0693333, 0.0405,
	  0.0186667, 0.00483333, 0., 0.00516667, 0.0213333, 0.0493333, 0.0893333,
	  0.141333, 0.205333, 0.281333, 0.369333, 0.469333, 0.581333, 0.705333,
	  0.841333, 0.989333, 1.14933, 1.32133, 1.50533, 1.70133, 1.90933, 2.12933, 2.36133]

[@test_approx_eq_eps ch[index] evaluate(cr, [value]) 1.e-4
									for (index, value) in enumerate(seq)];


cr = ModifiedCressieRead(-.5, .2)

ch = [2., 0.935089, 0.611146, 0.40911, 0.270178, 0.171573, 0.101613, .0533599,
      0.0222912, 0.00526681, 0., 0.00476461, 0.0182195, 0.039449, 0.0682857,
      0.10473, 0.148781, 0.200439, 0.259705, 0.326578, 0.401058, 0.483146,
      0.572841, 0.670143, 0.775052, 0.887568, 1.00769, 1.13542, 1.27076,
      1.41371, 1.56426]
[@test_approx_eq_eps ch[index] evaluate(cr, [value]) 1.e-4
									for (index, value) in enumerate(seq)];


################################################################################
##
## Gradient - Modified Cressie Read
##
################################################################################

seq =0:.101:3
cr = ModifiedCressieRead(2., .2)


	ch = [-0.5, -0.494899, -0.479598, -0.454096, -0.418392, -0.372487, -0.316382,
	      -0.250075, -0.173568, -0.0868595, 0.01005, 0.117161, 0.2344, 0.3556,
	      0.4768, 0.598, 0.7192, 0.8404, 0.9616, 1.0828, 1.204, 1.3252, 1.4464,
	      1.5676, 1.6888, 1.81, 1.9312, 2.0524, 2.1736, 2.2948]
[@test_approx_eq_eps ch[index] gradient(cr, value) 1.e-4
									for (index, value) in enumerate(seq)];

cr = ModifiedCressieRead(-.5, .2)
ch = [-Inf, -4.29317, -2.44994, -1.63336, -1.14658, -0.81439, -0.569175,
      -0.378594, -0.224971, -0.0977226, 0.00992562, 0.102539, 0.183387,
      0.26022, 0.337053, 0.413887, 0.49072, 0.567553, 0.644387, 0.72122,
      0.798053, 0.874887, 0.95172, 1.02855, 1.10539, 1.18222, 1.25905,
      1.33589, 1.41272, 1.48955]
[@test_approx_eq_eps ch[index] gradient(cr, value) 1.e-4
									for (index, value) in enumerate(seq)];


################################################################################
##
## Hessian - Modified Cressie Read
##
################################################################################

cr = ModifiedCressieRead(3., .2)
ch = [0., 0.010201, 0.040804, 0.091809, 0.163216, 0.255025, 0.367236, 0.499849,
      0.652864, 0.826281, 1.0201, 1.23432, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44,
      1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44]
[@test_approx_eq_eps ch[index] hessian(cr, value) 1.e-4
									for (index, value) in enumerate(seq)];


cr = ModifiedCressieRead(-1/3., .2)
ch = [+Inf, 21.2604, 8.4372, 4.91371, 3.3483, 2.48663, 1.95001, 1.58772,
      1.32878, 1.13566, 0.986821, 0.869056, 0.784197, 0.784197, 0.784197,
      0.784197, 0.784197, 0.784197, 0.784197, 0.784197, 0.784197, 0.784197,
      0.784197, 0.784197, 0.784197, 0.784197, 0.784197, 0.784197, 0.784197, 0.784197]
[@test_approx_eq_eps ch[index] hessian(cr, value) 1.e-4
									for (index, value) in enumerate(seq)];






#############################################################################
##
## Kullback Leibler
##
#############################################################################
seq =0:.1:3
kl = KullbackLeibler()
ch = [1.0, 0.669741, 0.478112, 0.338808, 0.233484, 0.153426, .0935046, .0503275,
	  .0214852, 0.00517554, 0., 0.0048412, 0.0187859, 0.0410735, 0.0710611, 0.108198,
	  0.152006, 0.202068, 0.258016, 0.319522, 0.386294, 0.458068, 0.534606, 0.615691,
	  0.701125, 0.790727, 0.88433, 0.98178, 1.08293, 1.18766, 1.29584]
[@test_approx_eq_eps ch[index] evaluate(kl, [value]) 1.e-4
									for (index, value) in enumerate(seq)];


ch = [-Inf, -2.30259, -1.60944, -1.20397, -0.916291, -0.693147, -0.510826,
	  -0.356675, -0.223144, -0.105361, 0., 0.0953102, 0.182322, 0.262364,
	  0.336472, 0.405465, 0.470004, 0.530628, 0.587787, 0.641854, 0.693147,
	  0.741937, 0.788457, 0.832909, 0.875469, 0.916291, 0.955511, 0.993252,
	  1.02962, 1.06471, 1.09861]

[@test_approx_eq_eps ch[index] gradient(kl, value) 1.e-4
									for (index, value) in enumerate(seq)];


ch = [+Inf, 10., 5., 3.33333, 2.5, 2., 1.66667, 1.42857, 1.25,
	  1.11111, 1., 0.909091, 0.833333, 0.769231, 0.714286, 0.666667, 0.625,
	  0.588235, 0.555556, 0.526316, 0.5, 0.47619, 0.454545, 0.434783,
	  0.416667, 0.4, 0.384615, 0.37037, 0.357143, 0.344828, 0.333333]

[@test_approx_eq_eps ch[index] hessian(kl, value) 1.e-4
									for (index, value) in enumerate(seq)];

mkl = ModifiedKullbackLeibler(.2)
ch = [+Inf, 0.669741, 0.478112, 0.338808, 0.233484, 0.153426, 0.0935046,
	  0.0503275, 0.0214852, 0.00517554, 0., 0.0048412, 0.0187859, 0.0411847,
	  0.0719168, 0.110982, 0.158381, 0.214113, 0.278179, 0.350578, 0.43131,
	  0.520375, 0.617774, 0.723506, 0.837572, 0.959971, 1.0907, 1.22977, 1.37717,
	  1.5329, 1.69696]

[@test_approx_eq_eps ch[index] evaluate(mkl, [value]) 1.e-4
									for (index, value) in enumerate(seq)];

seq =0:.11:3
ch = [-Inf,-2.20727,-1.51413,-1.10866,-0.820981,-0.597837,-0.415515,-0.261365,-0.127833,-0.0100503,0.0953102,0.190655,0.282322,0.373988,0.465655,0.557322,0.648988,0.740655,0.832322,0.923988,1.01565,1.10732,1.19899,1.29065,1.38232,1.47399,1.56565,1.65732]

[@test_approx_eq_eps ch[index] gradient(mkl, value) 1.e-4
									for (index, value) in enumerate(seq)];


ch = [+Inf, 9.09091,4.54545,3.0303,2.27273,1.81818,1.51515,1.2987,1.13636,1.0101,0.909091,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333,0.833333]

[@test_approx_eq_eps ch[index] hessian(mkl, value) 1.e-4
									for (index, value) in enumerate(seq)];



#############################################################################
##
## Reverse Kullback Leibler
##
#############################################################################

seq =0:.1:3
mrkl = ModifiedReverseKullbackLeibler(.2)

ch = [+Inf,1.40259,0.809438,0.503973,0.316291,0.193147,0.110826,0.0566749,0.0231436,0.00536052,0.,0.00468982,0.0176784,0.0378173,0.0649007,0.0989284,0.139901,0.187817,0.242678,0.304484,0.373234,0.448928,0.531567,0.621151,0.717678,0.821151,0.931567,1.04893,1.17323,1.30448,1.44268]

[@test_approx_eq_eps ch[index] evaluate(mrkl, [value]) 1.e-4
									for (index, value) in enumerate(seq)];


ch = [+Inf,-8.09091,-3.54545,-2.0303,-1.27273,-0.818182,-0.515152,-0.298701,-0.136364,-0.010101,0.0909091,0.173611,0.25,0.326389,0.402778,0.479167,0.555556,0.631944,0.708333,0.784722,0.861111,0.9375,1.01389,1.09028,1.16667,1.24306,1.31944,1.39583]

ch = [+Inf,82.6446,20.6612,9.18274,5.16529,3.30579,2.29568,1.68663,1.29132,1.0203,0.826446,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444]


#############################################################################
##
## Fully Modified Kullback Leibler
##
#############################################################################

seq =-3:.1:3
mrkl = FullyModifiedReverseKullbackLeibler(.1, 1)

# \begin{array}{cc}
#  \{ &
# \begin{array}{cc}
#  50. (z-0.1)^2-9. (z-0.1)+1.40259 & z\leq 0.1 \\
#  z-\log (z)-1 & z>0.1\land z<2 \\
#  \frac{1}{8} (z-2)^2+\frac{z-2}{2}+1-\log (2) & z\geq 2 \\
# \end{array}
#  \\
# \end{array}

ch =    [509.80258509299404,478.402585092994,448.00258509299397,418.60258509299405,
         390.202585092994,362.80258509299404,336.402585092994,311.00258509299397,
         286.60258509299405,263.202585092994,240.802585092994,219.402585092994,
         199.002585092994,179.60258509299405,161.202585092994,143.80258509299404,
         127.40258509299403,112.00258509299402,97.60258509299405,84.20258509299404,
         71.80258509299404,60.402585092994016,50.00258509299401,40.60258509299402,
         32.20258509299401,24.802585092994043,18.402585092994038,13.002585092994035,
         8.602585092994033,5.202585092994035,2.8025850929940455,1.4025850929940447,
         0.8094379124340996,0.5039728043259353,0.3162907318741546,0.1931471805599453,
         0.11082562376599059,0.05667494393873229,0.023143551314209698,0.005360515657826262,
         0.,0.004689820195675182,0.01767844320604539,0.03763573553250893,0.06352776337878718,
         0.09453489189183562,0.12999637075426462,0.1693717489378297,0.21221333509788132,
         0.25814611382760544,0.3068528194400547,0.35810281944005495,0.4118528194400547,
         0.46810281944005505,0.5268528194400549,0.5881028194400547,0.6518528194400551,
         0.7181028194400548,0.7868528194400551,0.858102819440055,0.9318528194400547]

[@test_approx_eq_eps ch[index] evaluate(mrkl, [value]) 1.e-4
									for (index, value) in enumerate(seq)];


ch = [-319., -309., -299., -289., -279., -269., -259., -249., -239., -229.,
      -219., -209., -199., -189., -179., -169., -159., -149., -139., -129.,
      -119., -109., -99., -89., -79., -69., -59., -49., -39., -29., -19.,
      -9., -4., -2.33333, -1.5, -1., -0.666667, -0.428571, -0.25,
      -0.111111, 0., 0.0909091, 0.166667, 0.230769, 0.285714, 0.333333,
      0.375, 0.411765, 0.444444, 0.473684, 0.5, 0.525, 0.55, 0.575, 0.6,
      0.625, 0.65, 0.675, 0.7, 0.725, 0.75]

[@test_approx_eq_eps ch[index] gradient(mrkl, value) 1.e-4
									for (index, value) in enumerate(seq)];



#############################################################################
##
## Both a and b
##
#############################################################################

a = .1
b = .2


@test evaluate(KullbackLeibler(), [a], [b]) == 0.030685281944005494
@test evaluate(ReverseKullbackLeibler(), [a], [b]) == 0.038629436111989046
@test evaluate(CressieRead(3), a, b) == 0.017708333333333333
@test evaluate(CressieRead(-3), a, b) == 0.06666666666666667
@test evaluate(Divergences.HD(), a, b) == 0.03431457505076194
@test evaluate(Divergences.ChiSquared(), a, b) == (a-b)^2/(2*b)

@test gradient(KullbackLeibler(), a, b) == log(a/b)
@test gradient(ReverseKullbackLeibler(), a, b) == 1.0-b/a
@test gradient(CressieRead(3), a, b) == ((a/b)^(3)-1)/3
@test gradient(CressieRead(-3), a, b) == ((a/b)^(-3)-1)/(-3)
@test gradient(Divergences.HD(), a, b) == ((a/b)^(-1/2)-1)/(-1/2)
@test gradient(Divergences.ChiSquared(), a, b) == a/b-1.0

@test hessian(Divergences.ChiSquared(), [a], [b]) == [1/b]
@test hessian(KullbackLeibler(), [a], [b]) == [1/a]
@test hessian(ReverseKullbackLeibler(), [a], [b]) == [b/a^2]

@test hessian(CressieRead(3), [a], [b]) ==    [( (a/b)^3 )/a]
@test hessian(CressieRead(1), [a], [b]) ==    [( (a/b)   )/a]
@test hessian(CressieRead(-1/2), [a], [b]) == [( (a/b)^(-.5) )/a]


@test_approx_eq evaluate(CressieRead(1), [a], [b])[1] evaluate(ChiSquared(), [a], [b])[1]
@test hessian(CressieRead(1), [a], [b])[1] == hessian(ChiSquared(), [a], [b])[1]
@test gradient(CressieRead(1), [a], [b])[1] == gradient(ChiSquared(), [a], [b])[1]


## Additional tests

@test CressieRead(1) === CressieRead(1.0)
@test CressieRead(-1/2) === HD()
@test_throws(AssertionError, CressieRead(-1))
@test_throws(AssertionError, ModifiedCressieRead(-1, -1))
@test_throws(AssertionError, ModifiedKullbackLeibler(-1))
@test_throws(AssertionError, ModifiedReverseKullbackLeibler(-1))

@test_throws(AssertionError, FullyModifiedCressieRead(1, -1, 2))
@test_throws(AssertionError, FullyModifiedKullbackLeibler(-1, 2))
@test_throws(AssertionError, FullyModifiedReverseKullbackLeibler(-1, 2))

@test_throws(AssertionError, FullyModifiedCressieRead(1, .5, -2))
@test_throws(AssertionError, FullyModifiedKullbackLeibler(.5, -2))
@test_throws(AssertionError, FullyModifiedReverseKullbackLeibler(.5, -2))


@test_throws(DimensionMismatch, evaluate(KL(), rand(10), rand(11)))
@test_throws(DimensionMismatch, evaluate(RKL(), rand(10), rand(11)))
@test_throws(DimensionMismatch, evaluate(CR(1), rand(10), rand(11)))
@test_throws(DimensionMismatch, evaluate(ChiSquared(), rand(10), rand(11)))

@test_throws(DimensionMismatch, evaluate(MKL(1), rand(10), rand(11)))
@test_throws(DimensionMismatch, evaluate(MRKL(1), rand(10), rand(11)))
@test_throws(DimensionMismatch, evaluate(MCR(1, 1), rand(10), rand(11)))

@test_throws(DimensionMismatch, evaluate(FMKL(.5, .5), rand(10), rand(11)))
@test_throws(DimensionMismatch, evaluate(FMRKL(.5, .5), rand(10), rand(11)))
@test_throws(DimensionMismatch, evaluate(FMCR(1, .5, .5), rand(10), rand(11)))


## Test normalization

@test evaluate(KL(), [1.], [1.]) == 0.0
@test evaluate(RKL(), [1.], [1.]) == 0.0
@test evaluate(CR(2), [1.], [1.]) == 0.0
@test evaluate(CR(-3/2), [1.], [1.]) == 0.0

@test gradient(KL(), [1.], [1.]) == [0.0]
@test gradient(RKL(), [1.], [1.]) == [0.0]
@test gradient(CR(2), [1.], [1.]) == [0.0]
@test gradient(CR(-3/2), [1.], [1.]) == [0.0]

@test hessian(KL(), [1.], [1.]) == [1.0]
@test hessian(RKL(), [1.], [1.]) == [1.0]
@test hessian(CR(2), [1.], [1.]) == [1.0]
@test hessian(CR(-3/2), [1.], [1.]) == [1.0]

## Modified CR with both a, b

@test evaluate(MCR(2, .2), [.2], [.1]) == 0.05813333333333335
@test evaluate(MCR(2, .2), [.3], [.1]) == 0.23613333333333322
@test evaluate(MCR(2, .2), [.4], [.1]) == 0.5341333333333332

@test evaluate(MCR(-2, .2), [.2], [.1]) == 0.032407407407407406
@test evaluate(MCR(-2, .2), [.3], [.1]) == 0.12291666666666663
@test evaluate(MCR(-2, .2), [.4], [.1]) == 0.2712962962962963

@test evaluate(FMCR(2, .1, .2), [.2], [.1]) == 0.05813333333333335
@test evaluate(FMCR(2, .1, .2), [.3], [.1]) == 0.23613333333333322
@test evaluate(FMCR(2, .1, .2), [.4], [.1]) == 0.5341333333333332

@test evaluate(FMCR(-2, .1, .2), [.2], [.1]) == 0.032407407407407406
@test evaluate(FMCR(-2, .1, .2), [.3], [.1]) == 0.12291666666666663
@test evaluate(FMCR(-2, .1, .2), [.4], [.1]) == 0.2712962962962963

@test evaluate(FMCR(-2, .1, .2), [-.2], [.1]) == 231.29999999999995
@test evaluate(FMCR(-2, .1, .2), [-.3], [.1]) == 496.24999999999983
@test evaluate(FMCR(2, .1, .2), [-.2], [.1]) == 0.15435000000000001



@test gradient(FMCR(2, .1, .2), [.25], [.2]) == gradient(MCR(2, .2), [.25], [.2])
@test gradient(FMCR(-2, .1, .2), [.25], [.2]) == gradient(MCR(-2, .2), [.25], [.2])

@test_approx_eq gradient(FMCR(2, .1, .2), [-2.], [.2]) -1.505
@test_approx_eq gradient(FMCR(2, .1, .2), [.21], [.2]) 0.05125
@test_approx_eq gradient(FMCR(2, .1, .2), [2.], [.2]) 10.78

@test_approx_eq hessian(FMCR(2, .1, .2), [-2.], [.2]) 0.5
@test_approx_eq hessian(FMCR(2, .1, .2), [.21], [.2]) 5.25
@test_approx_eq hessian(FMCR(2, .1, .2), [2.], [.2]) 6

@test evaluate(MCR(-2, .1), [-.2], [.1]) == Inf
@test evaluate(KL(), [-.2], [.1]) == Inf
@test evaluate(RKL(), [-.2], [.1]) == Inf
@test evaluate(CR(1), [-.2], [.1]) == Inf
@test evaluate(ChiSquared(), [-.2], [.1]) == 0.4500000000000001


## Chi-squared
@test evaluate(ChiSquared(), .2, .1) == evaluate(ChiSquared(), [.2], [.1])
@test_approx_eq evaluate(ChiSquared(), .2, .1) evaluate(CR(1), [.2], [.1])
@test_approx_eq gradient(ChiSquared(), .2, .1) gradient(CR(1), [.2], [.1])
@test_approx_eq hessian(ChiSquared(), .2, .1) hessian(CR(1), [.2], [.1])
@test_approx_eq gradient(ChiSquared(), .2) gradient(CR(1), [.2])
@test_approx_eq hessian(ChiSquared(), .2) hessian(CR(1), [.2])
@test_approx_eq evaluate(ChiSquared(), [.2]) evaluate(CR(1), [.2])
@test_approx_eq gradient(ChiSquared(), [.2]) gradient(CR(1), [.2])
@test_approx_eq hessian(ChiSquared(), [.2]) hessian(CR(1), [.2])



