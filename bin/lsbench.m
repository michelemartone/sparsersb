# 
# Linear Solvers benchmarks using sparsersb.
# 
# TODO: this file shall host some linear system solution benchmarks using sparsersb.
# It may serve as a reference point when profiling sparsersb/librsb.
# Please note that sparsersb is optimized for large matrices.
#
1; # This is a script

function lsb_compare(A)
n=rows(A);
maxit = n;
b = ones (n, 1);
P = diag (diag (A));
[i,j,v]=find(sparse(A));
minres=1e-7;
printf("Solving a system of %d equations, %d nonzeroes.\n",n,nnz(A));

tic; Ao = sparse (i,j,v,n,n);obt=toc;
onz=nnz(Ao);
tic; [X, FLAG, RELRES, ITER] = gmres (Ao, b, [], minres, maxit, P); odt=toc;
cs="Octave   ";
onv=norm(Ao*X-b);
oRELRES=RELRES;
printf("%s took %.4f = %.4f + %.4f s and gave residual %g, flag %d, error norm %g.\n",cs,obt+odt,obt,odt,RELRES,FLAG,onv);

tic; Ar = sparsersb (i,j,v,n,n);rbt=toc;
#tic; Ar = sparsersb (Ao);rbt=toc;
rnz=nnz(Ar);
tic; [X, FLAG, RELRES, ITER] = gmres (Ar, b, [], minres, maxit, P); rdt=toc;
cs="sparsersb";
rnv=norm(Ar*X-b);
printf("%s took %.4f = %.4f + %.4f s and gave residual %g, flag %d, error norm %g.\n",cs,rbt+rdt,rbt,rdt,RELRES,FLAG,rnv);

if (onz != rnz)
	printf("Error: seems like matrices don't match: %d vs %d nonzeroes!\n",onz,rnz);
	quit(1);
else
end


if (RELRES>minres ) && (oRELRES<minres )
	printf("Error: sparsersb was not able to solve a system octave did (residuals: %g vs %g)!",RELRES,oRELRES);
	quit(1);
else
	printf("Both systems were solved, speedups for overall: %g, constructor: %g, iterations: %g.\n",(obt+odt)/(rbt+rdt),(obt)/(rbt),(odt)/(rdt));
end
	printf("\n");
end

# This one is based on what Carlo De Falco posted on the octave-dev mailing list:
# (he used n=1000, k=15)
n = 4;
k = 1; 
A= k * eye (n) + sprandn (n, n, .8);
lsb_compare(A);

n = 100;
k = 5; 
A= k * eye (n) + sprandn (n, n, .2);
lsb_compare(A);

n = 1000;
k = 15; 
A= k * eye (n) + sprandn (n, n, .2);
lsb_compare(A);

n = 5000;
k = 1500; 
A= k * eye (n) + sprandn (n, n, .2);
lsb_compare(A);

#nx=40,ny=20;
nx0=10;ny0=10;
for xm=1:2
#for ym=1:2
for ym=1:1
nx=nx0*(2^xm),ny=ny0*(2^ym);
hx=1/(nx+1);
hy=1/(ny+1);
# a symmetric example, from andreas stahel's notes on solving linear systems
Dxx=spdiags([ones(nx,1)-2*ones(nx,1) ones(nx,1)],[-1 0 1],nx,nx)/(hx^2);
Dyy=spdiags([ones(ny,1)-2*ones(ny,1) ones(ny,1)],[-1 0 1],ny,ny)/(hy^2);
A = - kron(Dxx, speye(ny))-kron(speye(nx),Dyy);
[xx,yy]=meshgrid(linspace(hx,1-hx,nx),linspace(ny,1-hy,ny));
b=xx(:);
lsb_compare(A);

# a symmetric example, from andreas stahel's notes on solving linear systems
Dy=spdiags([ones(ny,1) ones(ny,1)],[-1 1],ny,ny)/(2*hy);
A = - kron(Dxx, speye(ny))-kron(speye(nx),Dyy) + 5*kron(speye(nx),Dy);
[xx,yy]=meshgrid(linspace(hx,1-hx,nx),linspace(ny,1-hy,ny));
b=xx(:);
lsb_compare(A);
end
end

printf "All done."
quit(1);
