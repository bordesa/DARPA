#!/usr/bin/env python

from corpus_pb2 import *
import lxml.objectify as objectify
import lxml.etree as etree
import sys
from google.protobuf import descriptor
from optparse import OptionParser

def addNodes(el, fields):
	nodes = []
	for field in fields:
		nodes.append(addNode(el, field))		
	return nodes

def addNode(el, field):
	if field[0].label == descriptor.FieldDescriptor.LABEL_REPEATED:
		for value in field[1]:
			node = etree.SubElement(el, field[0].name)
			addNodes(node, value.ListFields())
		return node
	else:
		el[field[0].name] = field[1]
		return el

def openDB(filename):
	restaurants = RestaurantList()
	with open(filename, "rb") as input:
		data = "".join(input.readlines())
		restaurants.ParseFromString(data)
	return restaurants

def parseArgs():
	parser = OptionParser()
	
	parser.add_option("-i", "--input", dest="input",
		help="input protobuf file")
	
	parser.add_option("-o", "--output", dest="output",
		help="output XML file")
		
	options, args = parser.parse_args()
	return parser, options

if __name__ == "__main__":
	parser, options = parseArgs()
	if options.input == None or options.output == None:
		parser.print_help()
		exit(1)
	#endif
	
	print "reading DB..."
	restaurants = openDB(options.input)
	print "processing..."
	root = objectify.Element("OpenTable")
	for restaurant in restaurants.restaurant:
		restaurantNode = etree.SubElement(root, "restaurant", id=str(restaurant.id))
		addNodes(restaurantNode, restaurant.ListFields()[1:])
		
	with open(options.output, "wb") as output:
		output.write(etree.tostring(root, pretty_print=True))
	#endwith
#endif