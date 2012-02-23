#!/bin/bash
if test $# != 1 ; then echo "Please supply a single Matrix Market file name at the command line." ; exit; fi
matlab  -nojvm -nodisplay -nodesktop -nosplash -r "matlabbench('"$1"')"
