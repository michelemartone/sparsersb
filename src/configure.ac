#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.67])
AC_INIT([octave sparsersb package], [1.0.5])
dnl AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_LANG(C++)

AC_CHECK_TOOL([OCTAVE], [octave], [none])
AC_CHECK_TOOL([MKOCTFILE], [mkoctfile], [none])
AC_ARG_VAR([MKOCTFILE],[The mkoctfile executable])
AC_ARG_VAR([OCTAVE],[The octave executable])
AC_ARG_VAR([SPARSERSB_LDFLAGS],[Linking flags for sparsersb])
AC_ARG_VAR([SPARSERSB_CXXFLAGS],[C++ compilation flags for sparsersb])
AC_ARG_VAR([SPARSERSB_CXX11],[C++11 additional compilation flags for sparsersb])
AC_ARG_VAR([LIBRSB_TARBALL],[Environment variable equivalent for --with-librsb-tarball=..])

if [ test "$MKOCTFILE" = "none" ]; then
  AC_MSG_ERROR([mkoctfile required to install $PACKAGE_NAME])
fi

# Checks for librsb
have_rsb=no
librsb_conf='no'
librsb_conf_static='no' # internal

AC_ARG_WITH(static-librsb, 
	[  --with-static-librsb=[yes|no] Whether to use static linking with the librsb-conf script. (recommended: yes).],
	[ librsb_conf_static=$withval;
	 if x"$librsb_conf_static" != xyes -a x"$librsb_conf_static" != xno ; then
		AC_MSG_ERROR([--static-librsb=yes or --static-librsb=no !! ])
	 fi
	       	 ], [  ])

# BEGIN tarball-based on-the-fly install of librsb
AC_ARG_WITH(librsb-tarball, 
	[  --with-librsb-tarball=... Will use specified user provided librsb tarball (e.g. librsb-1.2.0.tar.gz) to build a local librsb installation. You can specify LIBRSB_CFLAGS to customize. With this option --with-librsb-conf will be ignored.],
	[ librsb_tarball=$withval ], [ librsb_tarball=no ])

if [ test x"$LIBRSB_TARBALL" != "x" ]; then
	AC_MSG_NOTICE([Detected user-set LIBRSB_TARBALL=... variable; equivalent to --with-librsb-tarball=... .])
        librsb_tarball="$LIBRSB_TARBALL"
else
	AC_MSG_NOTICE([No user-set LIBRSB_TARBALL environment variable detected (set it to build using a custom librsb tarball).])
fi

if test x"$librsb_tarball" != x"no" ; then
   	dnl AC_MSG_WARN([...])
	AC_MSG_NOTICE([Will create a local librsb build from user provided "$librsb_tarball" tarball. Will use default LIBRSB_CFLAGS.])
	test -f ${librsb_tarball} || AC_MSG_ERROR([Are you sure of having file $librsb_tarball ?])
	tar xzf "$librsb_tarball" || AC_MSG_ERROR([Error uncompressing "$librsb_tarball" ?])
	librsb_dir=`basename $librsb_tarball`
	librsb_dir=${librsb_dir/.tar.gz/}
	test -d "$librsb_dir" || AC_MSG_ERROR([Error accessing "$librsb_dir" -- are you sure you provided a valid librsb tarball ?])
	dnl LIBRSB_PREFIX="${LIBRSB_PREFIX:-`pwd`/local}"
	LIBRSB_PREFIX="`pwd`/local"
	cd "$librsb_dir" || AC_MSG_ERROR([Error entering "$librsb_dir" -- are you sure you provided a valid librsb tarball ?])
	#
	LIBRSB_CFLAGS="${LIBRSB_CFLAGS:--O3 -fPIC}"
	AC_MSG_NOTICE([Stepped in "$librsb_dir".])
	dnl AC_MSG_NOTICE([Configuring with LIBRSB_PREFIX=$LIBRSB_PREFIX ])
	AC_MSG_NOTICE([Configuring with LIBRSB_CFLAGS=$LIBRSB_CFLAGS ])
	./configure OCTAVE='false' CFLAGS="${LIBRSB_CFLAGS}" --prefix="${LIBRSB_PREFIX}" --disable-fortran-examples --disable-c-examples
	AC_MSG_NOTICE([Configured successfully.])
	make || AC_MSG_ERROR([Make step failed !])
	AC_MSG_NOTICE([Built successfully.])
	make install || AC_MSG_ERROR([Make step failed !])
	cd -
	librsb_conf="${LIBRSB_PREFIX}"/bin/librsb-config
	test -f ${librsb_conf} || AC_MSG_ERROR([Temporary librsb installation config file ${librsb_conf} not executable ? Something went wrong with the install ?!])
	AC_MSG_NOTICE([Temporary librsb installation in ${LIBRSB_PREFIX} done.])
	AC_MSG_NOTICE([Forcing static linking due to temporary librsb.])
	librsb_conf_static='yes'
else
	AC_MSG_NOTICE([No librsb tarball provided: following the default procedure.])
	true;
fi
#  END  tarball-based on-the-fly install of librsb

if test x"$librsb_conf" = x"no" ; then
AC_ARG_WITH(librsb-conf, 
	[  --with-librsb-conf=...          Path to the librsb-conf program which will be used to recover SPARSERSB_CXXFLAGS and SPARSERSB_LDFLAGS (unless specified explicitly).],
	[ librsb_conf=$withval ], [ librsb_conf=no ])
else
	AC_MSG_NOTICE([Using ${librsb_conf} config file from the local librsb installation.])
fi

if test x"$librsb_conf" != x"no" ; then
	AC_MSG_NOTICE([Using user provided $librsb_conf script for librsb.])
else
	dnl AC_MSG_NOTICE([Probing for a librsb-config script...])
	AC_CHECK_PROG([HAVE_LIBRSB_CONFIG], [librsb-config], [yes], [no])
	if [test "x$HAVE_LIBRSB_CONFIG" != x"no"]; then
		librsb_conf=librsb-config
	fi
fi

if [test x"$SPARSERSB_CXXFLAGS" != "x" ]; then
	AC_MSG_NOTICE([Using user set SPARSERSB_CXXFLAGS...])
else
	if [test x"$librsb_conf" != x"no"]; then
		AC_MSG_NOTICE([Setting SPARSERSB_CXXFLAGS from $librsb_conf --I_opts...])
		SPARSERSB_CXXFLAGS="`$librsb_conf --I_opts`"
	fi
fi

OCTAVE_CLI="$OCTAVE --no-gui --no-window-system";

SPARSERSB_USE_64BIT_IDX=`$OCTAVE_CLI --no-line-editing -qf --eval 'printf ("%i", sizemax() > intmax ("int32"))'`
if [test x"$SPARSERSB_USE_64BIT_IDX" = x"1" ]; then
	AC_MSG_NOTICE([Adding -D RSBOI_DETECTED_LONG_IDX to SPARSERSB_CXXFLAGS ...])
	SPARSERSB_CXXFLAGS="$SPARSERSB_CXXFLAGS -D RSBOI_DETECTED_LONG_IDX=1"
fi

if [test x"$SPARSERSB_LDFLAGS" != "x" ]; then
	AC_MSG_NOTICE([Using user set SPARSERSB_LDFLAGS...])
else
	if [test x"$librsb_conf" != x"no"]; then
	if [test x"$librsb_conf_static" = x"no"]; then
		AC_MSG_NOTICE([Setting SPARSERSB_LDFLAGS from --L_opts --libs])
		SPARSERSB_LDFLAGS="`$librsb_conf              --L_opts --libs`"
	else
		AC_MSG_NOTICE([Setting SPARSERSB_LDFLAGS from --static --libs --extra_libs])
		SPARSERSB_LDFLAGS="`$librsb_conf              --static --libs --extra_libs`"
	fi
	fi
fi

if [test x"$SPARSERSB_CXXFLAGS" = "x" ]; then
	AC_CHECK_HEADERS([rsb.h], [AC_SEARCH_LIBS([rsb_lib_init], [rsb], [SPARSERSB_CXXFLAGS=" "], [])], [] )
fi
if [test x"$SPARSERSB_LDFLAGS" = "x" ]; then
	AC_CHECK_HEADERS([rsb.h], [AC_SEARCH_LIBS([rsb_lib_init], [rsb], [SPARSERSB_LDFLAGS="-lrsb"], [])], [] )
fi

if [test x"$SPARSERSB_CXXFLAGS" = "x" ]; then
	AC_MSG_ERROR([$PACKAGE_NAME SPARSERSB_CXXFLAGS (librsb not detected)!])
fi
if [test x"$SPARSERSB_LDFLAGS" = "x" ]; then
	AC_MSG_ERROR([$PACKAGE_NAME requires SPARSERSB_LDFLAGS (librsb not detected)!])
fi
have_rsb=yes

AC_MSG_NOTICE([SPARSERSB_CXXFLAGS is $SPARSERSB_CXXFLAGS])
AC_MSG_NOTICE([SPARSERSB_LDFLAGS  is $SPARSERSB_LDFLAGS])

# TODO: alternative: build librsb with SPARSERSB_LIBRSB_TARBALL
# TODO: alternative: --with-librsb=tarball will skip all checks and use that tarball

if [ test "$have_rsb" != "yes" ]; then
	AC_MSG_ERROR([$PACKAGE_NAME requires librsb library])
fi

# check for octave functions
save_CXX="$CXX"
save_CXXFLAGS="$CXXFLAGS"
CXX=`${MKOCTFILE} -p CXX`
CXXFLAGS="$CXXFLAGS -I`$MKOCTFILE -p OCTINCLUDEDIR`"

# need to use interpreter->get_load_path in dev version of octave,
# prior to that methods of load_path were static
AC_CACHE_CHECK(
  [for octave_value function iscomplex],
  [octave_value_cv_iscomplex],
  [AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([
      #include <oct.h>
      #include <octave.h>
      ],
      [
        octave_value().iscomplex();
      ])],
    [octave_value_cv_iscomplex=yes],
    [octave_value_cv_iscomplex=no])
  ])
if test "$octave_value_cv_iscomplex" = "yes" ; then
  HAVE_OCTAVE_VALUE_ISCOMPLEX=-DHAVE_OCTAVE_VALUE_ISCOMPLEX
else
  HAVE_OCTAVE_VALUE_ISCOMPLEX=
fi
AC_SUBST(HAVE_OCTAVE_VALUE_ISCOMPLEX)

CC=$save_CXX
CXXFLAGS=$save_CXXFLAGS



AC_SUBST(OCTAVE)
AC_SUBST(TARGETS)
AC_SUBST(SPARSERSB_CXXFLAGS)
AC_SUBST(SPARSERSB_CXX11, ["-std=gnu++11"])
AC_SUBST(SPARSERSB_LDFLAGS)

AC_CONFIG_FILES([Makeconf])
AC_OUTPUT


AC_MSG_NOTICE([

$PACKAGE_NAME is configured with:
  SPARSERSB_LDFLAGS:     $SPARSERSB_LDFLAGS
  SPARSERSB_CXXFLAGS:    $SPARSERSB_CXXFLAGS
  SPARSERSB_CXX11:       $SPARSERSB_CXX11
  OCTAVE:                $OCTAVE
  MKOCTFILE:             $MKOCTFILE
dnl  LIBS:     $LIBS
dnl  CXXFLAGS: $CXXFLAGS $DEFS
dnl  TARGETS:  $TARGETS

You can build it with 'make' and after on, test with 'make tests'.
])
