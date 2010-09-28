import numpy, math, cPickle, sys
from linearutil import *
from numpy.random import shuffle

# 
def get_mse(preds, true_labels):
    if len(true_labels)>0:
        res=0
        for pp in zip(preds, true_labels):
            res+=(pp[0]-pp[1])**2
        return res/len(preds)
    return 1


def TrainAndOptimizeClassifer(TrainingData, ValidationData, verbose):

     #linesearch looking for the best C
    MAXSTEPS=10
    STEPFACTOR=10.
    INITIALC=0.001

    Ccurrent = INITIALC
    Cstepfactor = STEPFACTOR
    Cnew = Ccurrent * Cstepfactor

    C_to_allstats = {}
    Cbest = None
    Models= {}

    TrainingProblem = problem(TrainingData[0],TrainingData[1])

    print >> sys.stderr, "Performing line search to get the best C (%d steps)"% MAXSTEPS 
    while len(C_to_allstats) < MAXSTEPS:
        if Ccurrent not in C_to_allstats:
            # Compute the validation statistics for the current C
            param = '-c %f -s 0 -q'% Ccurrent
            m=train(TrainingProblem, param)
            preds, acc, probas = predict_online(ValidationData[0], ValidationData[1], m , '-b 1')
            C_to_allstats[Ccurrent] = get_mse(preds, ValidationData[0])
            Models[Ccurrent]=m
        if Cnew not in C_to_allstats:
            # Compute the validation statistics for the next C
            param = '-c %f -s 0 -q'% Cnew
            m=train(TrainingProblem, param)
            preds, acc, probas = predict_online(ValidationData[0], ValidationData[1], m , '-b 1')
            C_to_allstats[Cnew] = get_mse(preds, ValidationData[0])
            Models[Cnew]=m
          # If Cnew has a higher val mse than Ccurrent, then continue stepping in this direction
        if C_to_allstats[Cnew] < C_to_allstats[Ccurrent]:
            if verbose: 
                print >> sys.stderr, "\tvalmse[Cnew %f] = %f < valmse[Ccurrent %f] = %f" % (Cnew, C_to_allstats[Cnew], Ccurrent, C_to_allstats[Ccurrent])
            if Cbest is None or C_to_allstats[Cnew] < C_to_allstats[Cbest]:
                Cbest = Cnew
                if verbose: 
                    print >> sys.stderr, "\tNEW BEST: Cbest <= %f, valmse[Cbest] = %f" % (Cbest, C_to_allstats[Cbest])
            Ccurrent = Cnew
            Cnew *= Cstepfactor
            if verbose: 
                print >> sys.stderr, "\tPROCEED: Cstepfactor remains %f, Ccurrent is now %f, Cnew is now %f" % (Cstepfactor, Ccurrent, Cnew)
        # Else, reverse the direction and reduce the step size by sqrt.
        else:
            if verbose: 
                print >> sys.stderr, "\tvalmse[Cnew %f] = %f > valmse[Ccurrent %f] = %f" % (Cnew, C_to_allstats[Cnew], Ccurrent, C_to_allstats[Ccurrent])
            if Cbest is None or C_to_allstats[Ccurrent] < C_to_allstats[Cbest]:
                Cbest = Ccurrent
                if verbose: 
                    print >> sys.stderr, "\tCbest <= %f, valmse[Cbest] = %f" % (Cbest, C_to_allstats[Cbest])
            Cstepfactor = 1. / math.sqrt(Cstepfactor)
            Cnew = Ccurrent * Cstepfactor
            if verbose: 
                print >> sys.stderr, "\tREVERSE: Cstepfactor is now %f, Ccurrent remains %f, Cnew is now %f" % (Cstepfactor, Ccurrent, Cnew)

    allC = C_to_allstats.keys()
    allC.sort()
    if verbose: 
        for C in allC:
            print >> sys.stderr, "\tvalmse[C %f] = %f" % (C, C_to_allstats[C]),
            if C == Cbest: print >> sys.stderr, " *best*"
            else: print >> sys.stderr, ""
    else:
        print >> sys.stderr, "\tBestC %f with Validation MSE = %f" % (Cbest, C_to_allstats[Cbest])


    return Models[Cbest]




def Classifier(model, TestData, prefix):

    # perform predictions
    print >> sys.stderr, "Testing on %d ex"% len(TestData[1][1])   
    preds, acc, probas = predict_online(TestData[0], TestData[1], model , '-b 1')
    
    # compute mse (1 if no test labels given) and save predictions
    mse=get_mse(preds, TestData[0])
    print >> sys.stderr, "\tTest MSE = %f" % mse    
    pf=open(prefix+'.predictions', 'w')
    for pp in preds:
        pf.write(str(pp)+'\n')
    pf.close()



def loadTrainDataset(task, labelFile, vectorFile, trainIDXFile):
    
    prob_y=svm_read_problem_labels(labelFile)

    idx=dict()
    for line in open(trainIDXFile):
        idx[int(line.split()[0])]=True
    
    TrainData=[[],[]]
    ValidationData=[[],[vectorFile, dict()]]

    ex_cnt=0
    for line in open(vectorFile):
        if ex_cnt in idx:
            TrainData[0] += [prob_y[ex_cnt][task]]
            xi = {}
            for e in line.split():
                ind, val = e.split(":")
                xi[int(ind)] = float(val)
            TrainData[1] += [xi]
        else:
            ValidationData[0] += [prob_y[ex_cnt][task]]
            ValidationData[1][1][ex_cnt] = True
        ex_cnt+=1
        
    return TrainData, ValidationData


def loadTestDataset(task, labelFile, vectorFile):
    
    idx=dict()
    ex_cnt=0
    for line in open(vectorFile):
        idx[ex_cnt]=True
        ex_cnt+=1

    labels=[]
    if labelFile:
        assert(task!=None)
        prob_y=svm_read_problem_labels(labelFile)
        for i in prob_y:
            labels += [i[task]]
            
    return [labels, [vectorFile, idx]]


# parse command line arguments and call main function
if __name__ == "__main__":

    if len(sys.argv) < 6:
        print "Usage:", sys.argv[0], " TaskIndex TrainVectors TrainLabels TrainIndices TestVectors [TestLabels]"
        sys.exit(-1)
        
    TaskIndex = int(sys.argv[1])
    TrainVectors = sys.argv[2]
    TrainLabels = sys.argv[3]
    TrainIndices = sys.argv[4]
    TestVectors=sys.argv[5]
    TestLabels=None
    if len(sys.argv) ==7:
        TestLabels=sys.argv[6]

    TrainingData, ValidationData = loadTrainDataset(TaskIndex, TrainLabels, TrainVectors, TrainIndices)
    best_classifier = TrainAndOptimizeClassifer(TrainingData, ValidationData, True)
        
    TestData = loadTestDataset(TaskIndex, TestLabels, TestVectors)
    Classifier(best_classifier, TestData, ''+TestVectors[:TestVectors[3:].find('.')])
