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

# This is expected to work on author's machine.
SPARSERSB_VERNAME=sparsersb-1.0.1
SPARSERSB_ARCHIVES=~/src/sparsersb-archives
dist:
	sh autogen.sh
	rm -f $(SPARSERSB_ARCHIVES)/$(SPARSERSB_VERNAME).tar
	hg archive -t tar $(SPARSERSB_ARCHIVES)/$(SPARSERSB_VERNAME).tar  --exclude '*.hgignore' -X ".hg*"
	cd ../.. && tar rf $(SPARSERSB_ARCHIVES)/$(SPARSERSB_VERNAME).tar --add-file $(SPARSERSB_VERNAME)/src/configure
	cd ../.. && tar  f $(SPARSERSB_ARCHIVES)/$(SPARSERSB_VERNAME).tar \
	       	--delete $(SPARSERSB_VERNAME)/.hg_archival.txt --delete $(SPARSERSB_VERNAME)/.hgtags --delete $(SPARSERSB_VERNAME)/.hgignore
	gzip -f $(SPARSERSB_ARCHIVES)/$(SPARSERSB_VERNAME).tar
	tar tvzf $(SPARSERSB_ARCHIVES)/$(SPARSERSB_VERNAME).tar.gz
	cd && octave --no-gui --eval 'pkg -local -verbose -verbose  install $(SPARSERSB_VERNAME).tar.gz'
	cd && octave --no-gui --eval 'pkg load generate_html; generate_package_html ("sparsersb", "sparsersb-html", "octave-forge") ' && tar czf $(SPARSERSB_ARCHIVES)/sparsersb-html.tar.gz sparsersb-html
	gpg -sbav -u EF1258B8 $(SPARSERSB_ARCHIVES)/$(SPARSERSB_VERNAME).tar.gz 
	gpg -sbav -u EF1258B8 $(SPARSERSB_ARCHIVES)/sparsersb-html.tar.gz
	md5sum $(SPARSERSB_ARCHIVES)/$(SPARSERSB_VERNAME).tar.gz > $(SPARSERSB_ARCHIVES)/$(SPARSERSB_VERNAME).tar.gz.md5
	md5sum $(SPARSERSB_ARCHIVES)/sparsersb-html.tar.gz       > $(SPARSERSB_ARCHIVES)/sparsersb-html.tar.gz.md5
	ls -l $(SPARSERSB_ARCHIVES)/$(SPARSERSB_VERNAME).tar.gz* $(SPARSERSB_ARCHIVES)/sparsersb-html.tar.gz*

# This is expected to work on author's machine.
commit:
	cd .. ; hg commit -u michelemartone
	cd .. ; hg pull

doc: $(SPARSERSB_OCT)
	$(OCTAVE) -q --eval 'help sparsersb' | grep -v 'is a function from the'  > ../doc/sparsersb.txt
#

