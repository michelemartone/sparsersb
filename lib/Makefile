# There are two variables you can set in the environment to control this Makefile behaviour:
#
# 1. You may override with e.g. LIBRSB_PREFIX=`librsb-config --prefix`  and the entire build step will be skipped.
# 2. LIBRSB_CFLAGS 
#
LIBDIRBASENAME=librsb-1.2.0-rc4
LIBRSB_PREFIX?=$(shell pwd)/..
# LIBRSB_CFLAGS can be better set one level above.
LIBRSB_CFLAGS?=-O3
#
LIBRSB_RSB_H=$(LIBRSB_PREFIX)/include/rsb.h
#LIBRSB_CFLAGS?=-O3

$(LIBRSB_RSB_H):
	tar xzf ../var/$(LIBDIRBASENAME).tar.gz
	cd $(LIBDIRBASENAME) && \
		./configure CFLAGS="$(LIBRSB_CFLAGS)" --prefix=$(LIBRSB_PREFIX) --disable-fortran-examples --disable-c-examples && \
	       	nice make && \
	       	make install
	# cd librsb-1.2.0-rc4 && ./configure CFLAGS="$(LIBRSB_CFLAGS)" --enable-matrix-types=double,double\ complex --prefix=`pwd`/.. && nice make && make install
