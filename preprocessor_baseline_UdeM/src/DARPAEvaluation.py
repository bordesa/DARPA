import os
import sys

import numpy

from exp_scripts.DARPAscript import OpenTableSDAEexp, createvecfile
from ClassifierSmartMemory import TrainAndOptimizeClassifer, Classifier, loadTrainDataset, loadTestDataset

def SampleStratified(rand,ListLabel,ClassifierTrainingSize):
    """
    This function, given a numpy random generator, a list of labels and 
    a training size, will return a list of indexs of the desired size
    that match the label proportion of the complete label set
    """
    OutList = list(numpy.sort(list(set(ListLabel))))
    sorted_idx = numpy.argsort(ListLabel)
    NbLab = {}
    for i in OutList:
        NbLab.update({i:(numpy.sum(ListLabel== i))})
    samples = []
    idxct = 0
    for i in OutList:
        idx_class = sorted_idx[idxct: idxct + NbLab[i]]
        idx_class = idx_class[rand.permutation(len(idx_class))]
        samples += list(idx_class[: numpy.ceil(NbLab[i] / float(len(ListLabel)) * ClassifierTrainingSize)])
        idxct += NbLab[i]
    samples = numpy.asarray(samples)[rand.permutation(len(samples))]
    return list(samples[:ClassifierTrainingSize])

def SampleStratifiedFolds(ListLabel,FoldsNumber):
    """
    This function, given a list of labels and a number of fold
    will return a list of list of index (one for each fold) corresponding
    to the stratified folds.
    """
    sorted_idx = numpy.argsort(ListLabel)
    kfolds = []
    for i in range(FoldsNumber):
        kfolds += [list(numpy.asarray(sorted_idx)[i::FoldsNumber])]
    return kfolds

def ReadLabelFile(LabelFile, category):
    """
    This function returns a list of label corresponding to the category (0<=int<=5)
    """
    f = open(LabelFile, 'r')
    i = f.readline()
    listlab = []
    while i != '':
        listlabtmp = i[:-1].split(' ')
        listlab += [int(listlabtmp[category])]
        i = f.readline()
    return listlab
    
def CreateIdxFile(listidx,path):
    """
    This function creates an idx file from an idx list.
    """
    f = open(path,'w')
    for i in listidx:
        f.write('%s\n'%i)
    f.close()
        

def evalMain( FoldsNumber, ClassifierTrainingSize, DataPrefix, ModelPath, Seed ):
    numpy.random.seed(Seed)
    # learn the model
    #OpenTableSDAEexp('DARPA.conf',ModelPath)
    # createrepresentations
    #createvecfile(ModelPath+'/depth3',DataPrefix+ '-train_small.vec',3,ModelPath + '/DLrep_depth3_train.vec')
    #createvecfile(ModelPath+'/depth3',DataPrefix+ '-test_small.vec',3,ModelPath + '/DLrep_depth3_test.vec')
    #createvecfile(ModelPath+'/depth1',DataPrefix+ '-train_small.vec',1,ModelPath + '/DLrep_depth1_train.vec')
    #createvecfile(ModelPath+'/depth1',DataPrefix+ '-test_small.vec',1,ModelPath + '/DLrep_depth1_test.vec')

    resbaseline = {}
    resshallow = {}
    resdeep = {}

    for task in range(5):
        ListLabel = ReadLabelFile( DataPrefix + '-train_small.lab',task)
        Train_idx = SampleStratified(numpy.random,ListLabel,ClassifierTrainingSize)
        Folds_idx = SampleStratifiedFolds(list(numpy.asarray(ListLabel)[Train_idx]),FoldsNumber)
        resbaseline.update({task : []})
        resshallow.update({task : []})
        resdeep.update({task : []})
        for idxfold,k in enumerate(Folds_idx):
            # Creating the index file
            CreateIdxFile(list(numpy.asarray(Train_idx)[k]),DataPrefix + 'current_idx_train.idx')
            # baseline
            TrainingData, ValidationData = loadTrainDataset(task, DataPrefix+ '-train_small.lab', DataPrefix+ '-train_small.vec', DataPrefix + 'current_idx_train.idx')
            best_classifier = TrainAndOptimizeClassifer(TrainingData, ValidationData, True)
            TestData = loadTestDataset(task, DataPrefix+ '-test_small.lab', DataPrefix+ '-test_small.vec')
            resbaseline[task] += [Classifier(best_classifier, TestData, DataPrefix + 'baseline_task_%s_fold_%s'%(task,idxfold))]
            # Shallow
            TrainingData, ValidationData = loadTrainDataset(task, DataPrefix+ '-train_small.lab', ModelPath + '/DLrep_depth1_train.vec', DataPrefix + 'current_idx_train.idx')
            best_classifier = TrainAndOptimizeClassifer(TrainingData, ValidationData, True)
            TestData = loadTestDataset(task, DataPrefix+ '-test_small.lab', ModelPath + '/DLrep_depth1_test.vec')
            resshallow[task] += [Classifier(best_classifier, TestData, DataPrefix + 'baseline_task_%s_fold_%s'%(task,idxfold))]
            # Deep
            TrainingData, ValidationData = loadTrainDataset(task, DataPrefix+ '-train_small.lab', ModelPath + '/DLrep_depth3_train.vec', DataPrefix + 'current_idx_train.idx')
            best_classifier = TrainAndOptimizeClassifer(TrainingData, ValidationData, True)
            TestData = loadTestDataset(task, DataPrefix+ '-test_small.lab', ModelPath + '/DLrep_depth3_test.vec')
            resdeep[task] += [Classifier(best_classifier, TestData, DataPrefix + 'baseline_task_%s_fold_%s'%(task,idxfold))]

    for i in range(5):
        print >> sys.stderr, 'baseline', numpy.mean(resbaseline[i])
        print >> sys.stderr, 'shallow', numpy.mean(reshallow[i])
        print >> sys.stderr, 'deep', numpy.mean(resdeep[i])

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print "Usage:", sys.argv[0], "FoldsNumber ClassifierTrainingSize DataPrefix ModelPath Seed"
        sys.exit(-1)
    
    FoldsNumber = int(sys.argv[1])
    ClassifierTrainingSize = int(sys.argv[2])
    DataPrefix = sys.argv[3]
    ModelPath = sys.argv[4]
    Seed = int(sys.argv[5])
    evalMain ( FoldsNumber, ClassifierTrainingSize, DataPrefix, ModelPath, Seed )
    

