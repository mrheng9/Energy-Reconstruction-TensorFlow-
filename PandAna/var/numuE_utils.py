import numpy as np

from PandAna.utils.enums import *
from PandAna.utils.misc import GetPeriod

# For the Numu Energy Estimator
class SplineFit():
  def __init__(self, spline_spec):
    self.spline = spline_spec
    self.nstitch = len(self.spline)//2 - 1

    self.slopes = self.spline[3::2]
    self.x0 = self.spline[2::2]

    self.intercepts = [0]*self.nstitch
    self.InitIntercept()

  def InitIntercept(self):
    prev_intercept = self.spline[0]
    prev_slope = self.spline[1]
    for i in range(self.nstitch):
      self.intercepts[i] = prev_intercept + (prev_slope-self.slopes[i])*self.x0[i]
      prev_intercept = self.intercepts[i]
      prev_slope = self.slopes[i]
  
  def __call__(self, var):
    if var <= 0.:
      return 0.
    stitchpos = np.where(list(map(lambda i: i <= var, self.x0)))[0]
    if len(stitchpos):
      return self.slopes[stitchpos[-1]]*var + self.intercepts[stitchpos[-1]]
    else:
      return self.spline[1]*var + self.spline[0]
    
kSplineProd5MuonFDp1 = SplineFit([
  1.333815966663624564e-01, 2.006655289624899308e-01,
  8.500671131829244942e+00, 2.154209380453118716e-01,
  2.035090204717860729e+01, 6.660211866896596611e+00,
  2.037655171956283340e+01, 0.000000000000000000e+00
])

kSplineProd5HadFDp1 = SplineFit([
  1.085140541354344575e-02, 2.003132845468682977e+00,
  5.249999999978114396e-01, 1.587909262160043244e+00,
  8.547074710510785822e-01, 2.070213642469894921e+00
])

kSplineProd5MuonFDp2 = SplineFit([
  1.333041428039729304e-01, 2.010374129825994727e-01,
  9.015314992888956880e+00, 2.172102602545122607e-01,
  1.881250004932426734e+01, 1.939648844865335120e-01
])

kSplineProd5HadFDp2 = SplineFit([
  3.384568670664851731e-02, 1.916508023623156864e+00,
  2.749999891254556461e-01, 2.284913434279694400e+00,
  4.495957896158719880e-01, 1.631687421408977157e+00,
  7.893618284090087034e-01, 2.015303688339076693e+00
])

kSplineProd5MuonFDfhc = SplineFit([
  1.412869875558434574e-01, 1.985202329476516148e-01,
  7.247665189483523562e+00, 2.144069735971011192e-01,
  2.218750000031716141e+01, 6.699485408121087782e-02
])

kSplineProd5HadFDfhc = SplineFit([
  5.767990231564357195e-02, 1.091963220147889491e+00,
  4.894474691585748438e-02, 2.031445922414648386e+00,
  5.142642860092461188e-01, 1.567915254401344383e+00,
  8.200421075858435049e-01, 2.016845013606002102e+00
])

kSplineProd5MuonFDrhc = SplineFit([
  1.245271319206025379e-01, 2.033997627592860902e-01,
  9.766311956246607195e+00, 2.180838285862531922e-01,
  2.003715340979164949e+01, 1.863267567727432683e-01,
  2.256004612234155360e+01, 4.754398422961426951e-02
])

kSplineProd5HadFDrhc = SplineFit([
  4.022415096001341617e-02, 2.011711823080491790e+00,
  4.199763458287808504e-01, 1.595097006634894843e+00,
  7.030242302962290690e-01, 2.148979944911536766e+00,
  1.293968553045185210e+00, 1.500071121804977814e+00
])


# near detector
kSplineProd5ActNDfhc = SplineFit([
  1.522067501417963542e-01, 1.935351432875078992e-01,
  3.534675721653096403e+00, 2.025064113727464976e-01,
  6.048717848712632517e+00, 2.086419146240798550e-01
])

kSplineProd5CatNDfhc = SplineFit([
  6.860056229074447398e-02, 1.021995188252620562e-01,
  1.466667613491428046e-01, 5.498842494606275277e-01,
  2.260114901099927298e+00, 1.411396843018650538e+00,
  2.313275230972585472e+00, 3.115156857428397208e-01
])

kSplineProd5HadNDfhc = SplineFit([
  5.049552462442885581e-02, 1.422732975320812443e+00,
  6.048754927389610181e-02, 2.709662443207628613e+00,
  1.015235485148796579e-01, 2.173545876693023349e+00,
  5.064530757547176520e-01, 1.725707450251668051e+00
])

kSplineProd5ActNDrhc = SplineFit([
  1.717171287078189390e-01, 1.853305227171077318e-01,
  2.502586270065958907e+00, 1.990563298599958286e-01,
  5.036450674404544081e+00, 2.083816760775504540e-01
])

kSplineProd5CatNDrhc = SplineFit([
  1.689154853867225192e-03, 5.492279050571418075e-01
])

kSplineProd5HadNDrhc = SplineFit
([
  4.676766851054844215e-02, 2.206317277398726073e+00,
  3.848300672745982309e-01, 1.593035140670105099e+00,
  6.819800276504310865e-01, 2.100597007299316310e+00,
  1.362679543056420250e+00, 1.417283364717454974e+00
])

def GetSpline(run, det, isRHC, comp):
  period = GetPeriod(run, det)
  if (period==1) and (det == detector.kFD):
    return {"muon" : kSplineProd5MuonFDp1, "had" : kSplineProd5HadFDp1}[comp]
  if (period==2) and (det == detector.kFD):
    return {"muon" : kSplineProd5MuonFDp2, "had" : kSplineProd5HadFDp2}[comp]
  if (period in [3, 5, 9, 10]) and (det == detector.kFD):
    return {"muon" : kSplineProd5MuonFDfhc, "had" : kSplineProd5HadFDfhc}[comp]
  if (period in [4, 6, 7, 8]) and (det == detector.kFD):
    return {"muon" : kSplineProd5MuonFDrhc, "had" : kSplineProd5HadFDrhc}[comp]
  if (det == detector.kND) and isRHC:
    return {"act" : kSplineProd5ActNDrhc, "cat": kSplineProd5CatNDrhc, "had": kSplineProd5HadNDrhc}[comp]
  if (det == detector.kND) and not isRHC:
    return {"act" : kSplineProd5ActNDfhc, "cat": kSplineProd5CatNDfhc, "had": kSplineProd5HadNDfhc}[comp]
