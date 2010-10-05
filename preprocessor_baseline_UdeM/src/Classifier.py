import numpy, math, cPickle, sys
from linearutil import *
from numpy.random import shuffle

# 
def get_rmse(preds, true_labels):
    """
    get_rmse(preds, true_labels): -> RMSE
    
    Compute the RMSE given the list of predictions 
    and the true labels.
    """
    if len(true_labels)>0:
        res=0
        for pp in zip(preds, true_labels):
            res+=(pp[0]-pp[1])**2
        return math.sqrt(res/len(preds))
    return 1


def TrainAndOptimizeClassifer(TrainingData, ValidationData, verbose):
    """
    TrainAndOptimizeClassifer(TrainingData, ValidationData, verbose) -> Classifier_model

    Taking as input training data (stored in memory) and validation
    data (stored in memory), this function performs a line search to
    determine the best SVM C parameter and returns the best svm model
    (i.e. the one with the lowest validation RMSE).
    """
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

    print >> sys.stderr, "Training on %d ex"% len(TrainingData[0])
    print >> sys.stderr, "Validating on %d ex"% len(ValidationData[0])

    print >> sys.stderr, "Performing line search to get the best C (%d steps)"% MAXSTEPS 
    while len(C_to_allstats) < MAXSTEPS:
        if Ccurrent not in C_to_allstats:
            # Compute the validation statistics for the current C
            param = '-c %f -s 0 -q'% Ccurrent
            m=train(TrainingProblem, param)
            preds, acc, probas = predict(ValidationData[0], ValidationData[1], m , '-b 1')
            C_to_allstats[Ccurrent] = get_rmse(preds, ValidationData[0])
            Models[Ccurrent]=m
        if Cnew not in C_to_allstats:
            # Compute the validation statistics for the next C
            param = '-c %f -s 0 -q'% Cnew
            m=train(TrainingProblem, param)
            preds, acc, probas = predict(ValidationData[0], ValidationData[1], m , '-b 1')
            C_to_allstats[Cnew] = get_rmse(preds, ValidationData[0])
            Models[Cnew]=m
          # If Cnew has a higher val rmse than Ccurrent, then continue stepping in this direction
        if C_to_allstats[Cnew] < C_to_allstats[Ccurrent]:
            if verbose: 
                print >> sys.stderr, "\tvalrmse[Cnew %f] = %f < valrmse[Ccurrent %f] = %f" % (Cnew, C_to_allstats[Cnew], Ccurrent, C_to_allstats[Ccurrent])
            if Cbest is None or C_to_allstats[Cnew] < C_to_allstats[Cbest]:
                Cbest = Cnew
                if verbose: 
                    print >> sys.stderr, "\tNEW BEST: Cbest <= %f, valrmse[Cbest] = %f" % (Cbest, C_to_allstats[Cbest])
            Ccurrent = Cnew
            Cnew *= Cstepfactor
            if verbose: 
                print >> sys.stderr, "\tPROCEED: Cstepfactor remains %f, Ccurrent is now %f, Cnew is now %f" % (Cstepfactor, Ccurrent, Cnew)
        # Else, reverse the direction and reduce the step size by sqrt.
        else:
            if verbose: 
                print >> sys.stderr, "\tvalrmse[Cnew %f] = %f > valrmse[Ccurrent %f] = %f" % (Cnew, C_to_allstats[Cnew], Ccurrent, C_to_allstats[Ccurrent])
            if Cbest is None or C_to_allstats[Ccurrent] < C_to_allstats[Cbest]:
                Cbest = Ccurrent
                if verbose: 
                    print >> sys.stderr, "\tCbest <= %f, valrmse[Cbest] = %f" % (Cbest, C_to_allstats[Cbest])
            Cstepfactor = 1. / math.sqrt(Cstepfactor)
            Cnew = Ccurrent * Cstepfactor
            if verbose: 
                print >> sys.stderr, "\tREVERSE: Cstepfactor is now %f, Ccurrent remains %f, Cnew is now %f" % (Cstepfactor, Ccurrent, Cnew)

    allC = C_to_allstats.keys()
    allC.sort()
    if verbose: 
        for C in allC:
            print >> sys.stderr, "\tvalrmse[C %f] = %f" % (C, C_to_allstats[C]),
            if C == Cbest: print >> sys.stderr, " *best*"
            else: print >> sys.stderr, ""
    else:
        print >> sys.stderr, "\tBestC %f with Validation RMSE = %f" % (Cbest, C_to_allstats[Cbest])


    return Models[Cbest]




def Classifier(model, TestData, prefix):
    """
    Classifier(model, TestData, prefix) -> None
    
    Taking as input a classifier model and test data (stored in
    memory), this function performs the model prediction for each test
    example and saves them into a file termed after the provided
    prefix.
    """
    # perform predictions
    print >> sys.stderr, "Testing on %d ex"% len(TestData[1])   
    preds, acc, probas = predict(TestData[0], TestData[1], model , '-b 1')
    
    # compute rmse (1 if no test labels given) and save predictions
    rmse=get_rmse(preds, TestData[0])
    print >> sys.stderr, "\tTest RMSE = %f" % rmse    
    pf=open(prefix+'.predictions', 'w')
    for pp in preds:
        pf.write(str(pp)+'\n')
    pf.close()



def loadTrainDataset(task, labelFile, vectorFile, trainIDXFile):
    """
    loadTrainDataset(task, labelFile, vectorFile, trainIDXFile) -> Training Data, Validation Data

    Taking as input a large data set (given as a filename for labels
    -- labelFile, and a filename for feature vectors -- vectorFile), a
    task identifier (for OpenTable this is an integer between 0 and 4
    as they are 5 rating per example) and a list of index
    (trainIDXFile), this function returns a training set (containing
    all the examples listed in trainIDXFile) labeled with the rating
    of the identified task and a validation set containing the
    remaining examples also labeled.
    """
    prob_y=svm_read_problem_labels(labelFile)
    prob_x=svm_read_problem_vectors(vectorFile)
    assert(len(prob_x)==len(prob_y))

    idx=dict()
    for line in open(trainIDXFile):
        idx[int(line.split()[0])]=True
    

    TrainData=[[],[]]
    ValidationData=[[],[]]
    for i in range(len(prob_x)):
        if prob_y[i][task]>=0: # do not test on unrated examples (label=-1) only valid for the "noise" task)
            if i in idx:
                TrainData[0] += [prob_y[i][task]]
                TrainData[1] += [prob_x[i]]
            else:
                ValidationData[0] += [prob_y[i][task]]
                ValidationData[1] += [prob_x[i]]

    return TrainData, ValidationData


def loadTestDataset(task, labelFile, vectorFile):
    """
    loadTestDataset(task, labelFile, vectorFile) -> Test Data

    Taking as input a data set (given as a filename for labels --
    labelFile, and a filename for feature vectors -- vectorFile) and a
    task identifier (for OpenTable this is an integer between 0 and 4
    as they are 5 rating per example), this function returns a test
    set. If no labelFile is given, an unlabeled test set is retruned.
    """       
    prob_x=svm_read_problem_vectors(vectorFile)
    if labelFile:
        assert(task!=None)
        prob_y=svm_read_problem_labels(labelFile)
        labels=[]
        nprob_x=[]
        for i in zip(prob_y, prob_x):
            if i[0][task]>=0: # do not test on unrated examples (label=-1) only valid for the "noise" task)            
                labels += [i[0][task]]
                nprob_x+=[i[1]]
        return [labels, nprob_x]
    else:                        
        return [[], prob_x]


# parse command line arguments and call main function that will load
# data and build train, validation and test set, train and optimize a
# classifier using the train and the validation sets and save its test
# predictions into a file.
#
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
    del TrainingData
    del ValidationData
       
    TestData = loadTestDataset(TaskIndex, TestLabels, TestVectors)
    Classifier(best_classifier, TestData, TestVectors.rpartition('/')[2].rpartition('.')[0]+'_task'+str(TaskIndex))
