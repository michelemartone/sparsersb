#sinclude Makeconf
#
# You have to define RSBLIBDIR in the environment!
# You have to define RSBINCDIR in the environment!
#
#LIBRSB_CFLAGS?=-O0
LIBRSB_CFLAGS?=-O3
LIBRSB_CXXFLAGS?=-I$(RSBINCDIR)
#RSBLIBDIR?=/usr/local/lib
RSBLIBDIRDEFAULT=`pwd`/../lib/
RSBLIBDIR?=$(RSBLIBDIRDEFAULT)
LIBRSB_LIBS?=$(RSBLIBDIR)/librsb.a
#
#
#

.PHONY: all check

SPARSERSB_OCT=sparsersb.oct
all: $(SPARSERSB_OCT)

tags: *.cc
	ctags *

#include omf/Makefile.inc
#LIBRSB_CXXFLAGS=-I$(RSBINCDIR)
#LIBRSB_LIBS=$(RSBLIBDIR)/librsb.a
library:
	if test "$(RSBLIBDIR)" = "$(RSBLIBDIRDEFAULT)" ; then \
		make -C $(RSBLIBDIR) LIBRSB_CFLAGS=$(LIBRSB_CFLAGS) ; fi

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

