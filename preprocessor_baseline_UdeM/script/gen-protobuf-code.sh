#!/bin/sh
PROJECTROOT=`dirname $0`/..
PROTOC=$PROJECTROOT/lib/install/bin/protoc

$PROTOC -I $PROJECTROOT/src/ --python_out=$PROJECTROOT/src/ $PROJECTROOT/src/OpenTable.proto
