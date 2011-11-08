sinclude Makeconf

.PHONY: all check

SPARSERSB_OCT=sparsersb.oct
all: $(SPARSERSB_OCT)

tags: *.cc
	ctags *

#include omf/Makefile.inc
#LIBRSB_CXXFLAGS=-I$(RSBINCDIR)
#LIBRSB_LIBS=$(RSBLIBDIR)/librsb.a

$(SPARSERSB_OCT): sparsersb.cc
	$(MKOCTFILE) -D''RSB_SPARSERSB_LABEL=sparsersb $(LIBRSB_CXXFLAGS) -o $@ $< $(LIBRSB_LIBS)

rtest: $(SPARSERSB_OCT)
	$(OCTAVE) --silent rsb.m
	$(OCTAVE) --silent sparsetester.m

bench: $(SPARSERSB_OCT)
	$(OCTAVE) --silent obench.m
	
tests: rtest
check: rtest

clean:
	-$(RM) *.o octave-core core *.oct *~ tags

# This is expected to work on author's machine.
dist:
	echo "warning: we are making a tarball out of SVN repositories: NOT this checked out copy."
	rm -fR $(PACKAGE_NAME)
	svn export `svn info  | grep URL | sed 's/^.* //;s/src$$//'g` $(PACKAGE_NAME)
	cd $(PACKAGE_NAME)/src && sh autogen.sh && cd -
	rm -fR $(PACKAGE_NAME)/src/old
	rm -fR $(PACKAGE_NAME)/src/TODO.txt
	rm -fR $(PACKAGE_NAME)/src/oldjunk
	tar czf $(PACKAGE_NAME).tgz $(PACKAGE_NAME)
	tar tzf $(PACKAGE_NAME).tgz

doc: $(SPARSERSB_OCT)
	$(OCTAVE) -q --eval 'help sparsersb' | grep -v 'is a function from the'  > ../doc/sparsersb.txt
