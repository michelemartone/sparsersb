# 
#  Copyright (C) 2011-2017   Michele Martone   <michelemartone _AT_ users.sourceforge.net>
# 
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses/>.
# 
#
# This program shall attempt solution of a problem saved in the MATLAB 
#  format as for the University of Florida collection.
#
# One with a problem structured as in e.g.:
#  http://www.cise.ufl.edu/research/sparse/mat/Hamm/memplus.mat
#  http://www.cise.ufl.edu/research/sparse/mat/Schenk_ISEI/barrier2-9.mat
# 
# s=load("~/barrier2-9.mat");
1; # This is a script

if length(getenv("SPARSERSB_TEST")) == 0 ; pkg load sparsersb ; end

disp " ***********************************************************************"
disp "**                    A usage example of sparsersb.                    **"
disp "** You can supply 'sparsersb' matrices to iterative method routines.   **"
disp "** If the matrix is large enough, this shall secure good performance   **"
disp "** of matrix-vector multiply: up to you to find method+linear system ! **"
disp "** p.s.: Invoke 'demo sparsersb' to get just a first working overview. **"
disp " ***********************************************************************"

s=load(argv(){length(argv())});
n=rows(s.Problem.A);
minres=1e-3;
#maxit = n;
maxit = 100;
b=s.Problem.b;

oct_A=sparse(s.Problem.A);
rsb_A=sparsersb(s.Problem.A);

printf (" **** Loaded a %d x %d matrix with %.3e nonzeroes ****\n", n, columns(s.Problem.A), nnz(s.Problem.A) );

X0=[];
RELRES=2*minres;
TOTITER=0;
M1=[]; M2=[];
M1=sparse(diag(s.Problem.A)\ones(n,1));
M2=sparse(diag(ones(n,1)));

nsb=str2num(sparsersb(rsb_A,"get","RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T"));
printf (" **** The 'sparsersb' matrix consists of %d RSB blocks. ****\n", nsb);

disp " *********** Invoking pcg using a 'sparse'    matrix ******************* ";
tic; [X1, FLAG, RELRES, ITER] = pcg (oct_A, b, minres, maxit, M1,M2,X0); odt=toc;
toc

disp " *********** Invoking pcg using a 'sparsersb' matrix ******************* ";
tic; [X1, FLAG, RELRES, ITER] = pcg (rsb_A, b, minres, maxit, M1,M2,X0); odt=toc;
toc

disp " ** Attempting autotuning 'sparsersb' matrix (pays off on the long run * ";
tic; rsb_A=sparsersb(rsb_A,"autotune","n",1);
toc;
nsb=str2num(sparsersb(rsb_A,"get","RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T"));
printf (" **** The 'sparsersb' matrix consists of %d RSB blocks now. **** \n", nsb);

disp " ****** Invoking pcg using a 'sparsersb' matrix (might be faster now) *** ";
tic; [X1, FLAG, RELRES, ITER] = pcg (rsb_A, b, minres, maxit, M1,M2,X0); odt=toc;
toc

disp " *********************************************************************** ";
quit(1);
