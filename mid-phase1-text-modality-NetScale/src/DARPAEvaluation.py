import cPickle
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
        if i != -1:
            idx_class = sorted_idx[idxct: idxct + NbLab[i]]
            idx_class = idx_class[rand.permutation(len(idx_class))]
            if -1 in NbLab.keys():
                samples += list(idx_class[: numpy.ceil(NbLab[i] / float(len(ListLabel)-NbLab[-1]) * ClassifierTrainingSize)]) 
            else:
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
    OpenTableSDAEexp(ModelPath+'DARPA.conf',ModelPath)
    
    # createrepresentations
    createvecfile(ModelPath+'/depth3',DataPrefix+ '-test.vec',3,ModelPath + '/DLrep_depth3_test.vec')
    createvecfile(ModelPath+'/depth1',DataPrefix+ '-test.vec',1,ModelPath + '/DLrep_depth1_test.vec')

    resbaseline = {}
    resshallow = {}
    resdeep = {}

    for task in range(5):
        ListLabel = ReadLabelFile( DataPrefix + '-train.lab',task)
        Train_idx = SampleStratified(numpy.random,ListLabel,ClassifierTrainingSize)
        orig_vec = open(DataPrefix + '-train.vec','r').readlines()
        orig_lab = open(DataPrefix + '-train.lab','r').readlines()
        textvec = ''
        textlab = ''
        for idx in Train_idx:
            textvec += orig_vec[idx]
            textlab += orig_lab[idx]
        vec_10k = open(DataPrefix + '-train_10k_task%s.vec'%task,'w')
        lab_10k = open(DataPrefix + '-train_10k_task%s.lab'%task,'w')
        vec_10k.write(textvec)
        lab_10k.write(textlab)
        vec_10k.close()
        lab_10k.close()
        createvecfile(ModelPath+'/depth3',DataPrefix+ '-train_10k_task%s.vec'%task,3,ModelPath + '/DLrep_depth3_train_10k_task%s.vec'%task)
        createvecfile(ModelPath+'/depth1',DataPrefix+ '-train_10k_task%s.vec'%task,1,ModelPath + '/DLrep_depth1_train_10k_task%s.vec'%task)
        ListLabel = ReadLabelFile(DataPrefix + '-train_10k_task%s.lab'%task,task)
        Folds_idx = SampleStratifiedFolds(ListLabel,FoldsNumber)
        resbaseline.update({task : []})
        resshallow.update({task : []})
        resdeep.update({task : []})
        for idxfold,k in enumerate(Folds_idx):
            # Creating the index file
            CreateIdxFile(k,DataPrefix + '_current_idx_train.idx')
            # baseline
            TrainingData, ValidationData = loadTrainDataset(task, DataPrefix+ '-train_10k_task%s.lab'%task, DataPrefix+ '-train_10k_task%s.vec'%task, DataPrefix + '_current_idx_train.idx')
            best_classifier = TrainAndOptimizeClassifer(TrainingData, ValidationData, True)
            TestData = loadTestDataset(task, DataPrefix+ '-test.lab', DataPrefix+ '-test.vec')
            resbaseline[task] += [Classifier(best_classifier, TestData, DataPrefix + '_baseline_task_%s_fold_%s'%(task,idxfold))]
            # Shallow
            TrainingData, ValidationData = loadTrainDataset(task, DataPrefix+ '-train_10k_task%s.lab'%task, ModelPath + '/DLrep_depth1_train_10k_task%s.vec'%task, DataPrefix + '_current_idx_train.idx')
            best_classifier = TrainAndOptimizeClassifer(TrainingData, ValidationData, True)
            TestData = loadTestDataset(task, DataPrefix+ '-test.lab', ModelPath + '/DLrep_depth1_test.vec')
            resshallow[task] += [Classifier(best_classifier, TestData, DataPrefix + '_shallow_task_%s_fold_%s'%(task,idxfold))]
            # Deep
            TrainingData, ValidationData = loadTrainDataset(task, DataPrefix+ '-train_10k_task%s.lab'%task, ModelPath + '/DLrep_depth3_train_10k_task%s.vec'%task, DataPrefix + '_current_idx_train.idx')
            best_classifier = TrainAndOptimizeClassifer(TrainingData, ValidationData, True)
            TestData = loadTestDataset(task, DataPrefix+ '-test.lab', ModelPath + '/DLrep_depth3_test.vec')
            resdeep[task] += [Classifier(best_classifier, TestData, DataPrefix + '_deep_task_%s_fold_%s'%(task,idxfold))]
            
            f = open(ModelPath + 'kfold_results_dictionnaries.pkl','w')
            cPickle.dump(resbaseline,f,-1)
            cPickle.dump(resshallow,f,-1)
            cPickle.dump(resdeep,f,-1)
            f.close()

    for i in range(5):
        print >> sys.stderr, 'baseline', numpy.mean(resbaseline[i]), " +/- ", numpy.std(resbaseline[i])
        print >> sys.stderr, 'shallow', numpy.mean(resshallow[i]), " +/- ", numpy.std(resshallow[i])
        print >> sys.stderr, 'deep', numpy.mean(resdeep[i]), " +/- ", numpy.std(resdeep[i])

    f = open(ModelPath + 'kfold_results_dictionnaries.pkl','w')
    cPickle.dump(resbaseline,f,-1)
    cPickle.dump(resshallow,f,-1)
    cPickle.dump(resdeep,f,-1)
    f.close()

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
    

