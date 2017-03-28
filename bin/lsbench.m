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
# Linear Solvers benchmark demos using sparsersb.
# 
# TODO: this file shall host some linear system solution benchmarks using sparsersb.
# It may serve as a reference point when profiling sparsersb/librsb.
# Please note that sparsersb is optimized for large matrices.
#
1; # This is a script

disp " ***********************************************************************"
disp "**   Usage example of sparsersb solving linear systems with GMRES.     **"
disp "** Matrices large enough for 'sparsersb' to likely outperform 'sparse'.**"
disp "** p.s.: Invoke 'demo sparsersb' to get just a first working overview. **"
disp " ***********************************************************************"

function lsb_compare(A)
n=rows(A);
maxit = n;
b = ones (n, 1);
P = diag (diag (A));
[i,j,v]=find(sparse(A));
minres=1e-7;
disp " ***********************************************************************"
printf("Solving a random system of %d equations, %d nonzeroes.\n",n,nnz(A));
disp " ***********************************************************************"

tic; Ao = sparse (i,j,v,n,n);obt=toc;
onz=nnz(Ao);
tic; [X, FLAG, RELRES, ITER] = gmres (Ao, b, [], minres, maxit, P); odt=toc;
cs="Octave   ";
onv=norm(Ao*X-b);
oRELRES=RELRES;
printf("%s took %.2e = %.2e + %.2e s and gave residual %g, flag %d, error norm %g.\n",cs,obt+odt,obt,odt,RELRES,FLAG,onv);

tic; Ar = sparsersb (i,j,v,n,n);rbt=toc;
#tic; Ar = sparsersb (Ao);rbt=toc;
rnz=nnz(Ar);
tic; [X, FLAG, RELRES, ITER] = gmres (Ar, b, [], minres, maxit, P); rdt=toc;
cs="sparsersb";
rnv=norm(Ar*X-b);
printf("%s took %.2e = %.2e + %.2e s and gave residual %g, flag %d, error norm %g.\n",cs,rbt+rdt,rbt,rdt,RELRES,FLAG,rnv);

if (onz != rnz)
	printf("Error: seems like matrices don't match: %d vs %d nonzeroes!\n",onz,rnz);
	quit(1);
else
end


if (RELRES>minres ) && (oRELRES<minres )
	printf("Error: sparsersb was not able to solve a system octave did (residuals: %g vs %g)!",RELRES,oRELRES);
	quit(1);
else
	iters=ITER(length(ITER));
	printf("Both systems were solved, overall speedup: %.1ex  (%.1es -> %.1es) \n  (matrix construction: %.1ex, %d iterations: %.1ex).\n",(obt+odt)/(rbt+rdt),(obt+odt),(rbt+rdt),(obt)/(rbt),iters,(odt)/(rdt));
	#if (obt+odt)/(rbt+rdt) > 1.0 
	#	printf("overall: %.1ex\n",(obt+odt)/(rbt+rdt));
	#end
end
	printf("\n");
end

# This one is based on what Carlo De Falco once posted on the octave-dev mailing list:
# (he used n=1000, k=15)

# Toy size.
#n = 4;
#k = 1; 
#A= sqrt(k) * eye (n) + sprandn (n, n, .9);
#lsb_compare(A);

# Toy size.
#n = 100;
#k = 5; 
#A= sqrt(k) * eye (n) + sprandn (n, n, .8);
#lsb_compare(A);

n = 2000;
k = 1000; 
A= sqrt(k) * eye (n) + sprandn (n, n, .4);
lsb_compare(A);

n = 5000;
k = 1500; 
A= sqrt(k) * eye (n) + sprandn (n, n, .2);
lsb_compare(A);

disp "All done."
disp "Notice how for large matrices the matrix construction is slower..."
disp "... but multiplications are faster !"
disp " ***********************************************************************"
