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
	$(OCTAVE) --silent ../bin/octavebench.m 
	$(OCTAVE) --norc -q --eval 'demo sparsersb'

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

SPARSERSB=sparsersb-1.0.2
ARCHIVE=$(HOME)/src/sparsersb-archives
GPGSIGNCMD=gpg -sbav -u 1DBB555AEA359B8AAF0C6B88E0E669C8EF1258B8 

help:
	@echo "# As a user, './configure ... .; make;' shall suffice."
	@echo "# If you are developing, you might be interested in 'make' followed by either of dist, sdist, sign, tarballs, html-doc."

dist: tarballs html-doc

sdist: sign tarballs html-doc

sign: tarballs html-doc
	$(GPGSIGNCMD) $(ARCHIVE)/$(SPARSERSB).tar.gz
	gpg --verify  $(ARCHIVE)/$(SPARSERSB).tar.gz.asc
	$(GPGSIGNCMD) $(ARCHIVE)/sparsersb-html.tar.gz
	gpg --verify  $(ARCHIVE)/sparsersb-html.tar.gz.asc

tarballs:
	rm -f $(ARCHIVE)/$(SPARSERSB).tar
	hg archive -t tar  $(ARCHIVE)/$(SPARSERSB).tar  --exclude '*.hgignore' -X ".hg*"
	rm -f ../../$(SPARSERSB)
	ln -s `pwd`/.. ../../$(SPARSERSB)
	cd ../.. && tar rf $(ARCHIVE)/$(SPARSERSB).tar --add-file $(SPARSERSB)/src/configure
	cd ../.. && tar  f $(ARCHIVE)/$(SPARSERSB).tar \
	       	        --delete $(SPARSERSB)/.hg_archival.txt --delete $(SPARSERSB)/.hgtags --delete $(SPARSERSB)/.hgignore
	#tar tvf $(SPARSERSB).tar
	gzip -f  $(ARCHIVE)/$(SPARSERSB).tar
	tar tvzf $(ARCHIVE)/$(SPARSERSB).tar.gz

html-doc:
	rm -fR   ./octfiles-tmp ./sparsersb-html $(SPARSERSB)
	mkdir -p ./octfiles-tmp 
	#tar xvzf $(ARCHIVE)/$(SPARSERSB).tar.gz
	# If missing.. pkg install -forge generate_html
	$(OCTAVE) --norc --eval 'pkg prefix ./octfiles-tmp ; echo on; diary sparsersb-diary-install.log; more off; pkg install -verbose '$(ARCHIVE)/$(SPARSERSB).tar.gz'; echo off; pkg load generate_html; generate_package_html ("sparsersb", "sparsersb-html", "octave-forge")'
	tar czf $(ARCHIVE)/sparsersb-html.tar.gz sparsersb-html
	echo "You maybe want to remove dirs: ./octfiles-tmp ./sparsersb-html $(SPARSERSB)"
	ls -l   $(ARCHIVE)/sparsersb-html.tar.gz $(ARCHIVE)/$(SPARSERSB).tar.gz 
	md5sum  $(ARCHIVE)/sparsersb-html.tar.gz $(ARCHIVE)/$(SPARSERSB).tar.gz 
	cd $(ARCHIVE)/
	
doc: $(SPARSERSB_OCT)
	$(OCTAVE) --norc -q --eval 'help sparsersb' | grep -v 'is a function from the'  > ../doc/sparsersb.txt

