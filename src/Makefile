#sinclude Makeconf
#
# Makeconf is disabled now.
# You can customize the build by modifying any of:
#
# 1. LIBRSB_PREFIX
# 2. LIBRSB_CXXFLAGS
#
LIBRSB_PREFIX?=$(shell pwd)/..
RSBINCDIR=$(LIBRSB_PREFIX)/include
RSBLIBDIR=$(LIBRSB_PREFIX)/lib
LIBRSB_CXX11?=-std=gnu++11
LIBRSB_CXXFLAGS?=-I$(RSBINCDIR) $(LIBRSB_CXX11)
LIBRSB_LIBS=`$(LIBRSB_PREFIX)/bin/librsb-config --static --extra_libs` `$(LIBRSB_PREFIX)/bin/librsb-config --libdir`/librsb.a
#

.PHONY: all check

SPARSERSB_OCT=sparsersb.oct
all: $(SPARSERSB_OCT)

tags: *.cc
	ctags *

library:
	if test "$(RSBLIBDIR)" = $(shell pwd)/../lib ; then \
	echo "Will build librsb so to have RSBINCDIR=$(RSBINCDIR) and RSBLIBDIR=$(RSBLIBDIR)"; \
		make -C $(RSBLIBDIR) ; fi

$(SPARSERSB_OCT): sparsersb.cc library
	CXXFLAGS="$(CXXFLAGS)" $(MKOCTFILE)  -D'RSB_SPARSERSB_LABEL=sparsersb' $(RSBOI_CXXFLAGS) $(LIBRSB_CXXFLAGS) -o $@ $< $(LIBRSB_LIBS) $(SPARSERSB_LIBS)

rtest: $(SPARSERSB_OCT)
	$(OCTAVE) --silent ../inst/sparsersbtester.m
	$(OCTAVE) --silent ../bin/octavebench.m ../bin/pd.mtx

bench: $(SPARSERSB_OCT)
	$(OCTAVE) --silent ../bin/obench.m
	$(OCTAVE) --silent ../bin/lsbench.m
	$(OCTAVE) --silent ../inst/sparsersbbench.m
	
tests: rtest
check: rtest

clean:
	-$(RM) *.o octave-core core *.oct *~ tags

# This is expected to work on author's machine.
dist_old:
	echo "warning: we are making a tarball out of SVN repositories: NOT this checked out copy."
	rm -fR $(PACKAGE_NAME)
	svn export `svn info  | grep URL | sed 's/^.* //;s/src$$//'g` $(PACKAGE_NAME)
	cd $(PACKAGE_NAME)/src && sh autogen.sh && cd -
	rm -fR $(PACKAGE_NAME)/src/old
	rm -fR $(PACKAGE_NAME)/src/TODO.txt
	rm -fR $(PACKAGE_NAME)/src/oldjunk
	tar czf $(PACKAGE_NAME).tgz $(PACKAGE_NAME)
	tar tzf $(PACKAGE_NAME).tgz

dist:
	hg archive --exclude '.hg*' ~/sparsersb-1.0.0.tar.gz

doc: $(SPARSERSB_OCT)
	$(OCTAVE) -q --eval 'help sparsersb' | grep -v 'is a function from the'  > ../doc/sparsersb.txt

