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
k = 1500; 
A = k * eye (n) + sprandn (n, n, .2);
b = ones (n, 1);
nnz (A)
P = diag (diag (A));
tic, [x, flag] = gmres (A, b, [], 1e-7, n, P); dt=toc
printf("Octave took %f s.\n",dt);

As = sparsersb (A);
tic, [x, flag] = gmres (As, b, [], 1e-7, n, P); dt=toc
printf("librsb took %f s.\n",dt);

