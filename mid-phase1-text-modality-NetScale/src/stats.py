#from corpus_pb2 import *
from OpenTable_pb2 import *
from glob import glob
from numpy import *
import sys

class OpenTableStats(object):
	def __init__(self):
		self.totalReviews = 0
		self.totalRestaurants = 0
		self.sizes = []
		
def collectStatsOnRestaurantList(restaurantList, stats):
	for restaurant in restaurantList.restaurant:
		stats.totalRestaurants += 1
		for review in restaurant.review:
			stats.sizes.append(len(review.text))
			stats.totalReviews += 1
		#endfor
	#endfor


def processDatabase(db, stats):
	print "processing \"%s\"..." % db
	with open (db) as input:
		data = "".join(input.readlines())
		restaurantList = RestaurantList()
		restaurantList.ParseFromString(data)
		collectStatsOnRestaurantList(restaurantList, stats)
		
def processDatabases(dbName):
	
	stats = OpenTableStats()
	
	for curDbName in dbName: 
		db = glob(curDbName)
		processDatabase(db[0], stats)
	
	print "mean number of characters per review: ", mean(stats.sizes)
	print "std deviation of characters per review: ", std(stats.sizes)
	print "minimum number of characters per review: ", min(stats.sizes)
	print "maximum number of characters per review: ", max(stats.sizes)
	print "total restaurants: ", stats.totalRestaurants
	print "total reviews: ", stats.totalReviews
	
if __name__ == "__main__":
	if len(sys.argv) < 1:
		print "Usage:", sys.argv[0], " dbName"
		print "dbName can contain wildcards"
		sys.exit(-1)

	processDatabases(sys.argv[1::])
	
