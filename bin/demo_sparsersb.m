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

if length(getenv("SPARSERSB_TEST")) == 0 ; pkg load sparsersb ; end

disp " ***********************************************************************"
disp "**                    A usage example of sparsersb.                    **"
disp "** A case large enough for 'sparsersb' to likely outperform 'sparse'.  **"
disp "** p.s.: Invoke 'demo sparsersb' to get just a first working overview. **"
disp " ***********************************************************************"

bs=100; # block size
bo=10; # block overlap
bc=700; # block count

# bs=2; # block size
# bo=1; # block overlap
# bc=2; # block count

nr=bs+(bc-1)*(bs-bo);
nc=nr;

disp "Constructing coefficients for a sparse diagonal blocks matrix."
printf ("Will use %d blocks each wide %d and overlapping %d.\n", bc, bs, bo);
ai=[];
aj=[];
av=[];
for i=1:bc
	# randomly generate block
	thr=0.3;
	b=rand(bs)+bc*eye(bs);
	b=b+b';
	[bi,bj,bv]=find(b>thr);
	io=(i-1)*(bs-bo); # i offset
	jo=(i-1)*(bs-bo); # j offset
	ai=[ai;bi+io];
	aj=[aj;bj+jo];
	av=[av;bv   ];
endfor

nz=length(av);
printf ("Obtained %.3e nonzeroes in a %d x %d matrix, average %.1e nonzeroes/row. \n", nz, nr, nc, nz/nr );

disp "Assembling a 'sparse'    matrix..."
tic;
ao=sparse(ai,aj,av);
printf ("Assembled 'sparse'    in %.1es.\n", toc);

disp "Assembling a 'sparsersb' matrix..."
tic;
ar=sparsersb(ai,aj,av);
nsb=str2num(sparsersb(ar,"get","RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T"));
printf ("Assembled 'sparsersb' in %.1es (%d RSB blocks).\n", toc, nsb);

# Uncomment to get more information:
# printf ("RSB matrix specific info: %s.\n", ds=sparsersb(ar,"get"));
# Uncomment the following to render the RSB blocks structure to a file.
# sparsersb(ar,"render","demo_sparsersb_matrix.eps")

maxt=4;

nrhs=1;
x=ones(nc,nrhs);
y=ones(nr,nrhs);

disp " ** Testing matrix-vector multiplication ********************************"

disp "Benchmarking 'sparse'    matrix-vector multiply..."
nt=0;tic;
while (toc < maxt)
	nt++;y+=ao*x;
end
ot=dt=toc;ot/=nt;
printf ("Performed %8d 'sparse'    matrix-vector multiplications in %.1es, %.2es each on average.\n", nt, dt, ot);

disp "Benchmarking 'sparsersb' matrix-vector multiply..."
nt=0;tic;
while (toc < maxt)
	nt++;y+=ar*x;
end
rt=dt=toc;rt/=nt;
printf ("Performed %8d 'sparsersb' matrix-vector multiplications in %.1es, %.2es each on average.\n", nt, dt, rt);
printf ("So 'sparsersb' is %.2ex times as fast as 'sparse'.\n", ot/rt);

disp "Attempting autotuning 'sparsersb' matrix..."
tic;
tr=sparsersb(ar,"autotune","n",nrhs);
nnb=str2num(sparsersb(tr,"get","RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T"));
dt=toc;
if ( nnb != nsb )
    printf ("Performed autotuning in   %.2es (%d -> %d RSB blocks).\n", dt, nsb, nnb);
    # Uncomment to get more information:
    # printf ("RSB matrix specific info: %s.\n", ds=sparsersb(tr,"get"));

    disp "Benchmarking 'sparsersb' matrix-matrix multiply..."
    nt=0;tic;
    while (toc < maxt)
    	nt++;y+=ar*x;
    end
    rt=dt=toc;rt/=nt;
    printf ("Performed %8d 'sparsersb' matrix-vector multiplications in %.1es, %.2es each on average.\n", nt, dt, rt);
    printf ("So 'sparsersb' is %.2ex times as fast as 'sparse'.\n", ot/rt);
else
    printf ("Autotuning procedure did not change the matrix.\n", dt, nsb, nnb);
end

disp " ** Testing matrix-matrix multiplication (rhs matrix is multi-vector) ***"

nrhs=5;
x=ones(nc,nrhs);
y=ones(nr,nrhs);

disp "Benchmarking 'sparse'    matrix-matrix multiply..."
nt=0;tic;
while (toc < maxt)
	nt++;y+=ao*x;
end
ot=dt=toc;ot/=nt;
printf ("Performed %8d 'sparse'    matrix-matrix multiplications in %.1es, %.2es each on average.\n", nt, dt, ot);

disp "Benchmarking 'sparsersb' matrix-matrix multiply..."
nt=0;tic;
while (toc < maxt)
	nt++;y+=ar*x;
end
rt=dt=toc;rt/=nt;
printf ("Performed %8d 'sparsersb' matrix-matrix multiplications in %.1es, %.2es each on average.\n", nt, dt, rt);
printf ("So 'sparsersb' is %.2ex times as fast as 'sparse'.\n", ot/rt);

disp "Attempting autotuning 'sparsersb' matrix..."
tic;
tr=sparsersb(ar,"autotune","n",nrhs);
nnb=str2num(sparsersb(tr,"get","RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T"));
dt=toc;
if ( nnb != nsb )
    printf ("Performed autotuning in   %.2es (%d -> %d RSB blocks).\n", dt, nsb, nnb);
    # Uncomment to get more information:
    # printf ("RSB matrix specific info: %s.\n", ds=sparsersb(tr,"get"));

    disp "Benchmarking 'sparsersb' matrix-matrix multiply..."
    nt=0;tic;
    while (toc < maxt)
    	nt++;y+=ar*x;
    end
    rt=dt=toc;rt/=nt;
    printf ("Performed %8d 'sparsersb' matrix-matrix multiplications in %.1es, %.2es each on average.\n", nt, dt, rt);
    printf ("So 'sparsersb' is %.2ex times as fast as 'sparse'.\n", ot/rt);
else
    printf ("Autotuning procedure did not change the matrix.\n", dt, nsb, nnb);
end

disp " ***********************************************************************"
disp "** You can adapt this benchmark to test your matrices so to check if   **"
disp "** they get multiplied faster with 'sparsersb' than with 'sparse'.     **"
disp " ***********************************************************************"
