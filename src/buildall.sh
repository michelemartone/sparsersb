#!/bin/sh
TMP=./tmp
LOCAL=`pwd`/$TMP/local
mkdir -p $TMP
mkdir -p $LOCAL
RSBDIR=$TMP/librsb
#svn co svn+ssh://nino/var/svn-repos/libmmvbr/libmmvbr $RSBDIR

cd $RSBDIR
#sh autogen.sh
#./configure --enable-optimize  --with-matrix-types=double --prefix=$LOCAL
#make
#make tests
#make install
cd -
echo RSBDIR_0=$LOCAL
make rtest RSBDIR_0=$LOCAL

