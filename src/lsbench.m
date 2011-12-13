# 
# Linear Solvers benchmarks using sparsersb.
# 
# TODO: this file shall host some linear system solution benchmarks using sparsersb.
# It may serve as a reference point when profiling sparsersb/librsb.
# Please note that sparsersb is optimized for large matrices.
#

# This one is based on what Carlo De Falco posted on the octave-dev mailing list:
# (he used n=1000, k=15)
n = 5000;
maxit = n;
k = 1500; 
A_= k * eye (n) + sprandn (n, n, .2);
[i,j,v]=find(sparse(A_));
tic, Ao = sparse (i,j,v,n,n);bt=toc;
b = ones (n, 1);
nnz (Ao)
P = diag (diag (Ao));
tic, [X, FLAG, RELRES, ITER] = gmres (Ao, b, [], 1e-7, maxit, P); dt=toc;
cs="Octave   ";
nv=norm(Ao*X-b);
printf("%s took %.4f = %.4f + %.4f s and gave residual %g, flag %d, error norm %g.\n",cs,bt+dt,bt,dt,RELRES,FLAG,nv);

#tic, Ar = sparsersb (Ao);bt=toc;
tic, Ar = sparsersb (i,j,v,n,n);bt=toc;
tic, [X, FLAG, RELRES, ITER] = gmres (Ar, b, [], 1e-7, maxit, P); dt=toc;
cs="sparsersb";
nv=norm(Ar*X-b);
printf("%s took %.4f = %.4f + %.4f s and gave residual %g, flag %d, error norm %g.\n",cs,bt+dt,bt,dt,RELRES,FLAG,nv);

