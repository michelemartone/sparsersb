
# shaped after http://octave.sourceforge.net/developers.html 

include ../../Makeconf

all: package

package:
	#test ! -f librsb && cd src && rename s/.m/.m.bak/g *.m && sh autogen.sh && ./configure
	test ! -f librsb && cd src && sh autogen.sh && ./configure
	#test   -f librsb && cd librsb && ./configure --prefix=`pwd`/local && make && cd ../src && sh autogen.sh && ./configure LIBRSB_CONFIG=

PKG_FILES = COPYING DESCRIPTION INDEX PKG_ADD $(wildcard src/*) \
        $(wildcard inst/*) $(wildcard doc/*) 

SUBDIRS = src/

.PHONY: $(SUBDIRS)

pre-pkg/%::
	test ! -f librsb && cd src && rename s/.m/.m.bak/g *.m

post-pkg/%::
	# Do nothing post packaging



