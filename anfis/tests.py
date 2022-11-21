import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy

ts = numpy.genfromtxt('../data/pulsar_data_train.csv', delimiter=',', skip_header=1,)#numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])
ts = ts[~numpy.any(numpy.isnan(ts), axis=1), :]

X = ts[:,0:-1]
Y = ts[:,-1]

mf = [
    [   ['gaussmf',{'mean':0.,'sigma':1.}],
        ['gaussmf',{'mean':1.,'sigma':1.}],
        ['gaussmf',{'mean':0.5,'sigma':1.}]],
        [   ['gaussmf',{'mean':0.,'sigma':1.}],
        ['gaussmf',{'mean':1.,'sigma':1.}],
        ['gaussmf',{'mean':0.5,'sigma':1.}]],
        [   ['gaussmf',{'mean':0.,'sigma':1.}],
        ['gaussmf',{'mean':1.,'sigma':1.}],
        ['gaussmf',{'mean':0.5,'sigma':1.}]],
        [   ['gaussmf',{'mean':0.,'sigma':1.}],
        ['gaussmf',{'mean':1.,'sigma':1.}],
        ['gaussmf',{'mean':0.5,'sigma':1.}]],
        [   ['gaussmf',{'mean':0.,'sigma':1.}],
        ['gaussmf',{'mean':1.,'sigma':1.}],
        ['gaussmf',{'mean':0.5,'sigma':1.}]],
        [   ['gaussmf',{'mean':0.,'sigma':1.}],
        ['gaussmf',{'mean':1.,'sigma':1.}],
        ['gaussmf',{'mean':0.5,'sigma':1.}]],
        [   ['gaussmf',{'mean':0.,'sigma':1.}],
        ['gaussmf',{'mean':1.,'sigma':1.}],
        ['gaussmf',{'mean':0.5,'sigma':1.}]],
[   ['gaussmf',{'mean':0.,'sigma':1.}],
        ['gaussmf',{'mean':1.,'sigma':1.}],
        ['gaussmf',{'mean':0.5,'sigma':1.}]],
]


mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=20)
print(round(anf.consequents[-1][0],6))
print(round(anf.consequents[-2][0],6))
print(round(anf.fittedValues[9][0],6))
if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
	print('test is good')

print("Plotting errors")
anf.plotErrors()
print("Plotting results")
anf.plotResults()
