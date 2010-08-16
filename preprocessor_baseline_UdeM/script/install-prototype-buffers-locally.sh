#!/bin/sh
set -e

PROJECTROOT=`dirname $0`/..
cd $PROJECTROOT/lib/protobuf

# make the C++
./configure --prefix=`pwd`/../install --exec-prefix=`pwd`/../install
make && make install

# make the Python
cd python
python setup.py build

# Link the python lib up with our code
cd ../../..
cd src
ln -s ../lib/protobuf/python/build/lib.*/google
