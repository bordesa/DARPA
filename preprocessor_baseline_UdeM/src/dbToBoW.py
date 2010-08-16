# dbToBoW.py
# converts a GPB restaurant database to BoW feature matrix
# arguments: dbName dictionaryOutFile dataOutFile dictionarySize
# currently outputs libSVM sparse matrix format
# should be easy to change write function for other data formats
# can change labelFromReview function to suit your needs

# UPDATE from Netscale team (require porterStemmer.py):
# Addition of a preprocessing module (lowercasing, punctuation
# removing, stemming and non-ascii words deletion)


from __future__ import with_statement
import fileinput, sys, re, os, operator, itertools
from OpenTable_pb2 import *


# CHANGE 1 HERE: add prepocessing fct
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


def updateCountsFromReview (countDict, rev):
    """
    Updates count dict based on a Review structure.
    """


    # CHANGE 2 HERE: run preprocessing
    text=preprocess(rev.text)

    for word in text.split():
        
        # CHANGE 3 HERE: remove non-ascii words
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
            countDict = updateCountsFromReview(countDict, curRev)
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
            countDict = updateCountsFromReview(dict(), curRev)
            
            # write label for this file (each file is one example)
            curLabel = getLabelFromReviw(curRev)
            dataFile.write(curLabel)

            # iterate over wordlist so feature indices are in order            
            for ind, curWord in enumerate(wordList):
                if curWord in countDict:
                    # using binary presence, not counts
                    dataFile.write(" %d:1" % ind)
                    # if you want to use word counts comment above and uncomment below
                    #dataFile.write(" %d:%d" % (ind, countDict[curWord]))

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
    # get top numFeat words
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
