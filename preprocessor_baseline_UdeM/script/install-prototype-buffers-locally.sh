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
python setup.py test
mkdir -p ../../install//lib/python2.5/site-packages/
PYTHONPATH=$PYTHONPATH:../../install/lib/python2.5/site-packages/ python setup.py install --prefix=../../install/
python setup.py install --prefix=../../install/

# Link the python lib up with our code
cd ../../..
cd src
ln -s ../lib/protobuf/python/build/lib.*/google
