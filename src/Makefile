sinclude Makeconf
# Makeconf is created by the configure script.
# You can also not use it but set explicitly the following:
CXXFLAGS=$(SPARSERSB_CXXFLAGS)
LFLAGS=$(shell $(MKOCTFILE) -p LFLAGS) $(SPARSERSB_LDFLAGS)
CXXFLAGS_CXX11=$(SPARSERSB_CXX11)

.PHONY: all check

SPARSERSB_OCT=sparsersb.oct
all: $(SPARSERSB_OCT)

tags: *.cc
	ctags *

#library:
#	true # if test "$(RSBLIBDIR)" = $(shell pwd)/../lib ; then \
#	# echo "Will build librsb so to have RSBINCDIR=$(RSBINCDIR) and RSBLIBDIR=$(RSBLIBDIR)"; \
#		# make -C $(RSBLIBDIR) ; fi

$(SPARSERSB_OCT): sparsersb.cc
	LFLAGS="$(LFLAGS)" CXXFLAGS="$(CXXFLAGS) $(CXXFLAGS_CXX11)" $(MKOCTFILE) -D'RSB_SPARSERSB_LABEL=sparsersb' -o $@ $< 
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
#dist_old:
#	echo "warning: we are making a tarball out of SVN repositories: NOT this checked out copy."
#	rm -fR $(PACKAGE_NAME)
#	svn export `svn info  | grep URL | sed 's/^.* //;s/src$$//'g` $(PACKAGE_NAME)
#	cd $(PACKAGE_NAME)/src && sh autogen.sh && cd -
#	rm -fR $(PACKAGE_NAME)/src/old
#	rm -fR $(PACKAGE_NAME)/src/TODO.txt
#	rm -fR $(PACKAGE_NAME)/src/oldjunk
#	tar czf $(PACKAGE_NAME).tgz $(PACKAGE_NAME)
#	tar tzf $(PACKAGE_NAME).tgz

dist:
	rm -f ~/src/sparsersb-1.0.1.tar
	hg archive -t tar ~/src/sparsersb-1.0.1.tar  --exclude '*.hgignore' -X ".hg*"
	cd ../.. && tar rf ~/src/sparsersb-1.0.1.tar --add-file sparsersb-1.0.1/src/configure
	cd ../.. && tar  f ~/src/sparsersb-1.0.1.tar \
	       	--delete sparsersb-1.0.1/.hg_archival.txt --delete sparsersb-1.0.1/.hgtags --delete sparsersb-1.0.1/.hgignore
	#tar tvf ~/src/sparsersb-1.0.1.tar
	gzip -f ~/src/sparsersb-1.0.1.tar
	tar tvzf ~/src/sparsersb-1.0.1.tar.gz

doc: $(SPARSERSB_OCT)
	$(OCTAVE) -q --eval 'help sparsersb' | grep -v 'is a function from the'  > ../doc/sparsersb.txt

