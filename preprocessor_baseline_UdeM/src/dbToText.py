# dbToText.py 
# preprocess reviews and filter only the n mots frequent words and
# return them is a text format

from __future__ import with_statement
import fileinput, sys, re, os, operator, itertools
from OpenTable_pb2 import *


# prepocessing fct
import string, porterStemmer

def preprocess(revtext):

    # lowercasing
    text=revtext.lower()

    # removing punctuation
    for punct in string.punctuation:
        text = text.replace(punct,"")

    # no long space
    text=text.replace("     "," ") 
    text=text.replace("    "," ") 
    text=text.replace("   "," ") 
    text=text.replace("  "," ") 

    # stemming (with the porter Stemmer)
    text=porterStemmer.run(text)
    
    return text    


def updateCountsFromReview (countDict, rev, return_sentence):
    """
    Updates count dict based on a Review structure.
    """

    # run preprocessing
    text=preprocess(rev.text)
    # if you want to skip preprocessing, comment the line above and
    # uncomment the one below
    #text=rev.text


    sentence=[]
    for word in text.split():
        
        # remove non-ascii words
        try:
            word.encode('ascii')
        except UnicodeEncodeError:
            ()
        else:            
            # update count for this word
            newCount = -1
            if word in countDict:
                newCount = countDict[word]+1
            else:
                newCount = 1
            
            countDict.update({word : newCount})
            if return_sentence:
                sentence.append(word)
    if return_sentence:
        return sentence
    else:
        return countDict


def updateCountsFromRL( countDict, restaurantList ):
    """
    Traverses a restaurantList structure to build a dictionary.

    Cictionary stores counts for all words found in all the reviews in
    the restaurantList.
    """
    # for each restaurant
    for curRest in restaurantList.restaurant:
        # for each review
        for curRev in curRest.review:
            # use helper to update dict
            countDict = updateCountsFromReview(countDict, curRev, False)
    return countDict


def getLabelFromReviw (rev):
    """
    Returns a label string for a review.

    Change this to make a binary task, 
    or to specify a label besides overaall
    """

    return str(int(rev.overallRating))

def writeDataForRestaurants(dataFile, wordList, restaurants):
    """
    Writes all reviews in list of restaurants to file.

    - outputs libSVM format (label ind:vale ind:value ...)
    - outputs word presence, not count
    - uses getLabelFromReview to get label string for a review
    """
    # for each restaurant
    for curRest in restaurants:
        # for each review
        for curRev in curRest.review:
            # get counts for this review
            sentence = updateCountsFromReview(dict(), curRev, True)
            
            # write label for this file (each file is one example)
            curLabel = getLabelFromReviw(curRev)
            dataFile.write(curLabel)

            # iterate over wordlist so feature indices are in order            
            for curWord in enumerate(sentence):
                if curWord[1] in wordList:
                    dataFile.write(" %s" % curWord[1])

            dataFile.write("\n")
    return None


def gpbMain(dbFilenames, dictionaryOutFilename, dataOutFilename, numFeat):
    """
    Main function for BoW conversion for GPB database.

    Called after minimal command line argument processing.

    dbFilenames are the database files to open, in order, that
    constitute a full Restaurant List.
    """
    # full dict of counts 
    fullDictPtr = [dict()]
    # open output files
    dictFile = open(dictionaryOutFilename, 'w')
    dataFile = open(dataOutFilename, 'w')
    
    def processAndReturnRestaurants(dbName):
        """
        Returns a list of restaurants for each file in dbFilename
        """
        with open (dbName) as input:
            print "Openning and reading", dbName
            # read in the contents of the file and create a restaurantList
            data = "".join(input.readlines())
            restaurantList = RestaurantList()
            restaurantList.ParseFromString(data)
            
            fullDictPtr[0] = updateCountsFromRL( fullDictPtr[0], restaurantList )
            return restaurantList.restaurant

    fullDict = fullDictPtr[0]
    restaurantLists = [processAndReturnRestaurants(dbName) for dbName in dbFilenames]
    allRestaurants = list(itertools.chain(*restaurantLists))
    
    # sort dict by frequency
    sortedCountList = sorted(fullDict.items(), 
                             key=operator.itemgetter(1),                             
                             reverse=True)

    # get top numFeat words (if numFeat==0, get them all)
    if numFeat==0:
        numFeat=len(sortedCountList)
    topCountList = sortedCountList[0:numFeat]
    topWordList =  [o[0] for o in topCountList]
    
    
    # TODO preprocess the reviews only once in the process (currently 2 times) to save time..
    # write datafile for this dir
    writeDataForRestaurants(dataFile, topWordList, allRestaurants)
    
    print "Words dictionary size: ", len(sortedCountList)

    # write feature file
    for w in topWordList:
        dictFile.write("%s %s\n" %(w, fullDict[w]))
    
    # close output files
    dataFile.close()
    dictFile.close()


# parse command line arguments and call main function
if __name__ == "__main__":

    if len(sys.argv) < 5:
        print "Usage:", sys.argv[0], " dictionaryOutFile dataOutFile dictionarySize dbName1 [dbName2 [...]]"
        sys.exit(-1)

    dictionaryOutFile = sys.argv[1]
    dataOutFile = sys.argv[2]
    dictSize = sys.argv[3]
    dbNames = sys.argv[4:]

    gpbMain( dbNames, dictionaryOutFile, dataOutFile, int(dictSize))
