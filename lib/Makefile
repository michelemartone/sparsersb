LIBDIRBASENAME=librsb-1.2.0-rc4
LIBDIRPREFIX=$(PWD)/..

all: $(LIBDIRPREFIX)/librsb.a

$(LIBDIRPREFIX)/librsb.a:
	tar xzf ../var/$(LIBDIRBASENAME).tar.gz
	cd $(LIBDIRBASENAME) && \
		./configure CFLAGS=$(LIBRSB_CFLAGS) --prefix=$(LIBDIRPREFIX) --disable-fortran-examples --disable-c-examples && \
	       	nice make && \
	       	make install
	# cd librsb-1.2.0-rc4 && ./configure CFLAGS=$(LIBRSB_CFLAGS) --enable-matrix-types=double,double\ complex --prefix=`pwd`/.. && nice make && make install
