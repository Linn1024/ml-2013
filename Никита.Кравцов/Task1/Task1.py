class Vector:
    def __init__(self, data, label):
        self.data = data
        self.label = label

class Perceptron:
    def __init__(self, vector, steps):
        self.fun = numpy.zeros(vector[0].data.size)
        for i in range(steps):
            for j in vector:
                if (numpy.dot(self.fun.T, j.data)) != j.label:
                    self.fun += j.label * j.data


    def getTrain(self, vector):
        return numpy.count_nonzero([self.getLabel(v.data) - v.label] for v in vector) / len(vector)

    def getAc(self, vector):
        truePos = 0
        falsePos = 0
        falseNeg = 0
        trueNeg = 0
        for i in vector:
            if numpy.sign(numpy.dot(self.fun.T, i.data)) == i.label:
                if i.label == 1:
                    truePos += 1
                else:
                    trueNeg += 1
            else:
                if i.label == 1:
                    falseNeg += 1
                else:
                    falsePos += 1 
        return (truePos/(falsePos + truePos), (trueNeg + truePos)/(trueNeg + truePos  + falsePos + falseNeg))


def read_data(tp):
    FIN = open('wdbc.data')
    ss = FIN.readlines()
    vectors = numpy.array([Vector(numpy.array(list(map(float, s.split(',') [2:]))), 1 if s.split(',')[1] == 'M' else -1) for s in ss])
    perm = numpy.random.permutation(vectors.size)
    ts = int(tp * vectors.size)
    tr = perm[:ts]
    t = perm[ts:]
    FIN.close()

    return (vectors[tr], vectors[t])

train, test = read_data(0.3)
clas = Perceptron(train, 1024)

print("Training error = " + str(clas.get_training_error(test)))
ac = clas.getAc(test)
print("Precision = " + str(ac[0]))
print("Accuracy = " + str(ac[1]))