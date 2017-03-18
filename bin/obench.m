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
1; # This is a script

if length(getenv("SPARSERSB_TEST")) == 0 ; pkg load sparsersb ; end

disp " ***********************************************************************"
disp "**           A small 'sparse' vs 'sparsersb' benchmark.                **"
disp "**  On a few sample problems, tests:                                   **"
disp "**   - matrix-vector multiplication               (SPMV)               **"
disp "**   - matrix-vector multiplication transposed    (SPMV_T)             **"
disp "**   - sparse matrix-spares matrix multiplication (SPGEMM)             **"
disp "**   - and shows speedup ('RSB SPEEDUP' column)                        **"
disp "** p.s.: Invoke 'demo sparsersb' to get just a first working overview. **"
disp " ***********************************************************************"

disp "OP	ROWS	COLUMNS	NONZEROES	OPTIME	MFLOPS	RSB SPEEDUP	IMPLEMENTATION"

cmt="#";
#for n_=1:6*0+1
for n_=1:6
for ro=0:1
	n=n_*1000;
	m=k=n;
	# making vectors
	b=linspace(1,1,n)';
	ox=linspace(1,1,n)';
	bx=linspace(1,1,n)';
	# making matrices
	r=(rand(n)>.6);
	om=sparse(r);
	nz=nnz(om);
	M=10^6;
	if ro==1 
		printf("%s%s\n",cmt," reordering with colamd...");
		pct=-time; 
		p=colamd(om);
		pct+=time;
		pat=-time; 
		om=om(:,p);
		pat+=time;
		# TODO: use an array to select/specify the different reordering algorithms
		# printf("%g\t%g\t(%s)\n",(nz/M)/pct,(nz/M)/pat,"mflops for pct/pat");
		printf("# ...colamd took %.1es (%.1e nnz/s), ",pct,nz/pct);
		printf(   "  permutation took %.1es (%.1e nnz/s)\n",pat,nz/pat);
	else
		printf("%s%s\n",cmt," testing with no reordering");
	end
	#bm=sparsevbr(om);
	bm=sparsersb(sparse(om));
	#bm=sparsersb3(sparse(om));
	# stats
	flops=2*nz;
	## spmv
	ot=-time; ox=om*b; ot+=time;
	#
	bt=-time; bx=bm*b; bt+=time;
	t=ot; p=["octave-",version]; mflops=(flops/M)/t;
	printf("%s\t%d\t%d\t%d\t%.1es\t%g\t%.1ex\t%s\n","SPMV  ",m,k,nz,t,mflops,1    ,p);
	t=bt; p=["RSB"]; mflops=(flops/M)/t;
	printf("%s\t%d\t%d\t%d\t%.1es\t%g\t%.1ex\t%s\n","SPMV  ",m,k,nz,t,mflops,ot/bt,p);

	## spmvt
	ot=-time; ox=om.'*b; ot+=time;
	#
	bt=-time; bx=bm.'*b; bt+=time;
	t=ot; p=["octave-",version]; mflops=(flops/M)/t;
	printf("%s\t%d\t%d\t%d\t%.1es\t%g\t%.1ex\t%s\n","SPMV_T",m,k,nz,t,mflops,1    ,p);
	t=bt; p=["RSB"]; mflops=(flops/M)/t;
	printf("%s\t%d\t%d\t%d\t%.1es\t%g\t%.1ex\t%s\n","SPMV_T",m,k,nz,t,mflops,ot/bt,p);

	## spgemm
	ot=-time; ox=om*om; ot+=time;
	#
	bt=-time; bx=bm*bm; bt+=time;
	t=ot; p=["octave-",version]; 
	printf("%s\t%d\t%d\t%d\t%.1es\t\t%.1ex\t%s\n","SPGEMM",m,k,nz,t,1,    p);
	t=bt; p=["RSB"]; 
	printf("%s\t%d\t%d\t%d\t%.1es\t\t%.1ex\t%s\n","SPGEMM",m,k,nz,t,ot/bt,p);
endfor
endfor
disp " ***********************************************************************"
