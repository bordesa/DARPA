# splitDB.py
# this code splits the dataset into subsets
# these separate dbs can then be used for training/testing
# arguments: dbName perc1 perc2 .. percN
# generates N+1 db files that split dbName into the
# corresponding percentages
# each perc argument should be [1 - 99], and they should
# sum to < 100

from OpenTable_pb2 import *
import sys
from glob import glob
from numpy import *
from random import shuffle


def copyRestaurant(inRest, outRest):
    """
    copy restaurant inRest into blank restaurant outRest
    deep copy due to GPB design
    """
    outRest.id = inRest.id
    for inRev in inRest.review:
        outRev = outRest.review.add()
        outRev.overallRating = inRev.overallRating
        outRev.foodRating = inRev.foodRating
        outRev.ambianceRating = inRev.ambianceRating
        outRev.serviceRating = inRev.serviceRating
        outRev.noiseRating = inRev.noiseRating
        outRev.text = inRev.text
        
def numReviews (r):
    """
    returns the number of reviews for a given restaurant
# useful in map() calls
    """
    return size(r.review)

def splitNReviews(restaurantList, indList, numRevs):
    """
    Remove restaurants until we have at least numRevs reviews.

    Return as separate list, with sublist no longer present in
    original indices of items to remove are popped from indList new
    subset list, shortened index list, num in subset are returned
    """
    subList = RestaurantList()
    collectedRevs = 0
    # loop until we have enough reviews or nothing left
    while collectedRevs < numRevs and size(indList) > 0:
        # get next element from list
        curRest = restaurantList.restaurant[indList.pop()]
        # add it to sublist and update count
        copyRestaurant(curRest, subList.restaurant.add())
        collectedRevs = collectedRevs + numReviews(curRest)
        
    return subList, indList, collectedRevs



def splitDatabase(db, percArr):
    """
    Takes a db name string and creates split files percArr is an array
    of float values which sum to 1
    """
    # open the db and parse it
    with open (db) as input:
        data = "".join(input.readlines())
        restaurantList = RestaurantList()
        restaurantList.ParseFromString(data)
        
        # get total by summing all review counts
        totalRevs = map(numReviews, restaurantList.restaurant)
        totalRevs = array(map(int, totalRevs))
        totalRevs = totalRevs.sum()

        # random ordering of restaurant indices
        indList = range(1,len(restaurantList.restaurant))
        shuffle(indList)

        # make a new db for this split percent
        for p in percArr:
            newDBName = db + "."+ str(int(p*100)) + ".db"
            numRevs = int(totalRevs * p)
            print newDBName, ": ", numRevs, " ideal"

            # samples from list without replacement
            subList, indList, actualRevs = splitNReviews(restaurantList, indList, numRevs)
        
            print newDBName, ":  ", actualRevs, " actual"

            # write subset as new db
            f = open(newDBName, "wb")
            f.write(subList.SerializeToString())
            f.close()
    
		
def splitDatabasesByArgs():
    """
    The main function.  Takes the following command line arguments:

    dbName -- a libSVM features file output

    perc1, perc2, ... -- percentages, must sum to less than 100 ( the
       remainder will be split into a distinct group
    """
    if len(sys.argv) < 3:
        print "Usage:", sys.argv[0], " dbName perc1 perc2 .."
        sys.exit(-1)

    dbName = sys.argv[1]

    # create numpy array from percent arguments
    percList = map(float, sys.argv[2:])
    percList.append(0)
    percArr = array(percList)
    # bit of error checking
    if sum(percArr) > 99:
        print 'Error: percents given must sum to < 100. E.g. 25 50'
        exit(-1)

    percArr = percArr / 100        
    percArr[-1] = 1 - sum(percArr)
    print percArr, percArr.sum()

    # for each db name, split by the given percentages
    dbs = glob(dbName)
    for db in dbs:                         
        splitDatabase(db, percArr)
                
	
if __name__ == "__main__":
    splitDatabasesByArgs()

	
