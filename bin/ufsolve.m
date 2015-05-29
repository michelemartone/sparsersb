# 
#  Copyright (C) 2011-2015   Michele Martone   <michelemartone _AT_ users.sourceforge.net>
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
# This program shall attempt solution of a problem saved in the MATLAB format as for the University of Florida collection.
#
# e.g.: http://www.cise.ufl.edu/research/sparse/mat/Hamm/memplus.mat
# 
# s=load("~/memplus.mat");
1; # This is a script
s=load(argv(){length(argv())});
n=rows(s.Problem.A);
minres=1e-7;
maxit = n;
#maxit = 100;
b=s.Problem.b;
#A=sparse(s.Problem.A);
A=sparsersb(s.Problem.A);
X0=[];
RELRES=2*minres;
TOTITER=0;
M1=[]; M2=[];
M1=spdiag(A)\ones(n,1);
M2=spdiag(ones(n,1));
while RELRES >= minres ;
tic; [X1, FLAG, RELRES, ITER] = pcg (A, b, minres, maxit, M1,M2,X0); odt=toc;
RELRES
ITER;
TOTITER=TOTITER+ITER
toc
X0=X1;
end

