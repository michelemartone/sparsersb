## Copyright 2017 Julien Bect <jbect@users.sourceforge.net>
## Copyright 2015-2016 Carnë Draug
## Copyright 2015-2016 Oliver Heimlich
##
## Copying and distribution of this file, with or without modification,
## are permitted in any medium without royalty provided the copyright
## notice and this notice are preserved.  This file is offered as-is,
## without any warranty.

PACKAGE := $(shell grep "^Name: " DESCRIPTION | cut -f2 -d" ")
VERSION := $(shell grep "^Version: " DESCRIPTION | cut -f2 -d" ")

TARGET_DIR      := $(CURDIR)/target
RELEASE_DIR     := $(TARGET_DIR)/$(PACKAGE)-$(VERSION)
RELEASE_TARBALL := $(TARGET_DIR)/$(PACKAGE)-$(VERSION).tar.gz
HTML_DIR        := $(TARGET_DIR)/$(PACKAGE)-html
HTML_TARBALL    := $(TARGET_DIR)/$(PACKAGE)-html.tar.gz

HG_ID   := $(shell hg id --id | sed -e 's/+//')
HG_DATE := $(shell hg log --rev $(HG_ID) --template {date\|isodate})

# Follows the recommendations of https://reproducible-builds.org/docs/archives
# Note #1: GNU tar is assumed
# Note #2: --format=ustar selects the 'ustar' (POSIX.1-1988) tar format
define create_tarball
$(shell cd $(dir $(1)) \
    && find $(notdir $(1)) -print0 \
    | LC_ALL=C sort -z \
    | tar c --format=ustar --mtime="$(HG_DATE)" --mode=a+rX,u+w,go-w,ug-s \
            --owner=0 --group=0 --numeric-owner \
            --no-recursion --null -T - -f - \
    | gzip -9n > "$(2)")
endef

M_SOURCES   := $(wildcard inst/*.m) $(patsubst %.in,%,$(wildcard src/*.m.in))

OCTAVE ?= octave --no-window-system --silent

.PHONY: all help dist html release install build-inplace check run clean

all: build-inplace release

help:
	@echo "Targets:"
	@echo "   dist    - Create $(RELEASE_TARBALL) for release"
	@echo "   html    - Create $(HTML_TARBALL) for release"
	@echo "   release - Create both of the above and show md5sums"
	@echo
	@echo "   install - Install the package in GNU Octave"
	@echo "   check   - Execute package tests (w/o install)"
	@echo "   run     - Run Octave with development in PATH (no install)"
	@echo
	@echo "   clean   - Remove releases, html documentation"

%.tar.gz: %
	$(call create_tarball,$*,$@)

$(RELEASE_DIR): .hg/dirstate
	@echo "Creating package version $(VERSION) release ..."
	-$(RM) -r "$@"
	hg archive --exclude ".hg*" --type files "$@"
	cd "$@/src" && ./autogen.sh && $(RM) -r "autom4te.cache"

$(HTML_DIR): install
	@echo "Generating HTML documentation. This may take a while ..."
	-$(RM) -r "$@"
	cd src && $(OCTAVE) \
	  --eval "pkg load generate_html; " \
	  --eval 'generate_package_html ("${PACKAGE}", "$@", "octave-forge");'

dist: $(RELEASE_TARBALL)
html: $(HTML_TARBALL)

release: dist html
	md5sum $(RELEASE_TARBALL) $(HTML_TARBALL)
	@echo "Upload @ https://sourceforge.net/p/octave/package-releases/new/"
	@echo 'Execute: hg tag "release-${VERSION}" when the release is ready.'

install: $(RELEASE_TARBALL)
	@echo "Installing package locally ..."
	$(OCTAVE) --eval 'pkg ("install", "${RELEASE_TARBALL}")'


build-inplace: src/sparsersb.oct

src/configure: src/autogen.sh src/configure.ac
	cd src && ./autogen.sh

src/Makeconf: src/configure src/Makeconf.in
	cd src && ./configure

src/sparsersb.oct: src/Makefile src/Makeconf
	cd src && $(MAKE) sparsersb.oct


check: src/sparsersb.oct
	$(MAKE) -C src tests

run: src/sparsersb.oct
	$(OCTAVE) --persist --path "inst/" --path "src/"

clean:
	$(RM) -r $(TARGET_DIR)
	$(MAKE) -C src clean
	$(RM) -rf src/autom4te.cache
	$(RM) src/Makeconf src/config.log src/config.status src/configure
