#!/usr/bin/octave -q
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
# a benchmark program for octave/matlab

# TODO: fix output format 
# TODO: correct symmetric / hermitian matrices handling
# TODO: sound, time-and-runs-based benchmarking criteria 

if length(getenv("SPARSERSB_TEST")) == 0 ; pkg load sparsersb ; end

disp " ***********************************************************************"
disp "**           A small 'sparse' vs 'sparsersb' test / benchmark.         **"
disp "**      This is meant to be a demo, but not really an example.         **"
disp "**  You can invoke it supplying a Matrix Market matrix (e.g. pd.mtx).  **"
disp "**        Without arguments, will generate a test matrix.              **"
disp "** p.s.: Invoke 'demo sparsersb' to get just a first working overview. **"
disp " ***********************************************************************"

n=10;

function printbenchline(matrixname,opname,sw,times,nnz,tottime,mxxops,bpnz,msstr)
	printf("FIXME (temporary format)\n");
	printf("%s %s %s %d %d %.4f %10.2f %.4f %s\n",matrixname,opname,sw,times,nnz,tottime,mxxops,bpnz,msstr);
end

if nargin <= 0
# DGEMV benchmark
disp "**  Will generate a matrix...                                          **"
for o=1024:1024
#for o=1024:256:2048*2
	m=rand(o);
	v=linspace(1,1,o)';
	tic();
	for i=1:n; m*v; end
	t=toc();
	Mflops=n*2.0*nnz(m)/(10^6 * t);
	dgemvmflops=Mflops;
	printf("%d GEMV for order %d  in  %g secs, so %10f Mflops\n",n,o,t,n*2.0*o*o/(10^6 * t));
end
disp " ***********************************************************************"
else	# nargin > 0

# if nargin > 0, we continue
want_sparsersb_io=1;

if want_sparsersb_io != 1
	source("ext/mminfo.m");
	source("ext/mmread.m");
	source("ext/mmwrite.m");
end

#matrices=ls("*.mtx")';
f=1;
uc=2; # only 2 for the moment being.
while f<=nargin
	MB=1024*1024;
	printf("**  Will read Matrix Market matrix file %s ...\n",f);
	mmn=cell2mat(argv()(f))';
	mn=strtrim(mmn');
	tic();
	#nm=mmread(mn);
	if want_sparsersb_io == 1
		[nm,nrows,ncols,entries,rep,field,symm]=sparsersb(mn);
		nm=sparse(nm);
		if (symm=='S')
			uc+=1;
		end
	else
		[nm,nrows,ncols,entries,rep,field,symm]=mmread(mn);
		#if(symm=="symmetric")uc+=2;endif
		if(strcmp(symm,"symmetric"))uc+=1;endif
	end
	disp " "
	wr=0 ; # write rendering to file
	if wr==1 
		sparsersb(sparsersb(nm),"render",[mn,"-original.eps"]);
		pct=-time; 
		#p=colamd(nm);
		p=colperm(nm);
		pct+=time;
		pat=-time; 
		nm=nm(:,p);
		pat+=time;
		#sparsersb(sparsersb(nm),"render",[mn,"-colamd.eps"])
		sparsersb(sparsersb(nm),"render",[mn,"-colperm.eps"]);
	end
	fsz=stat(mn).size;
	rt=toc();
	[ia,ja,va]=find(nm);
	printf("%s: %.2f MBytes read in  %.4f s (%10.2f MB/s)\n",mn,fsz/MB,rt,fsz/(rt*MB));
	#ia=ia'; ja=ja'; va=va';
	sep=" ";
	csvlstring=sprintf("#mn entries nrows ncols");
	csvdstring=sprintf("%%:%s%s%d%s%d%s%d",mn,sep,entries,sep,nrows,sep,ncols);
for ski=1:uc
	oppnz=1;
	# FIXME: what about handling symmetry ?
	sparsekw="sparse";
	if(ski==2)sparsekw="sparsersb";endif
	if(ski==3);
		oppnz=2;
		sparsekw="sparsersb";
		tic(); [nm]=sparsersb(mn); rt=toc();
		sparsersb(nm,"info")
		printf("%s: %.2f MBytes read by librsb in  %.4f s (%10.2f MB/s)\n",mn,fsz/MB,rt,fsz/(rt*MB));
	endif
	if(ski==4);
		nm=tril(nm);
	endif
	[ia,ja,va]=find(nm);
	rnz=nnz(nm);
	printf(" *** Benchmarking '%s'.\n",sparsekw);
	# printf("symmetry ? %d\n",issymmetric(sparse(nm)));
	mrc=rows(nm); mcc=columns(nm);

	if(ski!=3);
	tic();
	eval(["for i=1:n;  om=",sparsekw,"(ia,ja,va,mrc,mcc,\"summation\"); end"]);
	printf(" *** Benchmarking '%s' instantiation from 'ia,ja,va'.\n",sparsekw);
	at=toc();
	#if(ski==2) tic(); nm=sparsersb(om,"autotune","N");om=nm; att=toc(); ;endif
	mnz=nnz(om);
	amflops=n*(mnz/(10^6 * at));
	printf("%s '%s' %d Instantiations for %d nnz in  %.4f secs, so %10.2f nnz/s\n",mn,sparsekw,n,rnz,at,amflops);
	else
	mnz=rnz;
	end

	if(ski==2)
		nsb=str2num(sparsersb(om,"get","RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T"));
		printf (" ** Assembled 'sparsersb' matrix has %d RSB blocks.\n", nsb);
	endif

	#rm=sparsersb(ia,ja,va);# UNFINISHED
	r=linspace(1,1,size(om,1))';
	v=linspace(1,1,size(om,2))';
	printf(" *** Benchmarking '%s' SPMV..\n",sparsekw);
	tic(); for i=1:n; r+=om  *v; end; umt=toc();
	UMflops=oppnz*n*2.0*mnz/(10^6 * umt);
	printf("%s '%s' %d SPMV  for %d nnz in  %.4f secs, so %10.2f Mflops\n",mn,sparsekw,n,mnz,umt, UMflops);
	bpnz=-1;  # FIXME: bytes per nonzero!
	msstr="?";# FIXME: matrix structure string!
	# FIXME: finish the following!
	#printbenchline(mn',"SPMV",sparsekw,n,mnz,umt, UMflops,bpnz,msstr);
	#
	tmp=r;r=v;v=tmp;
	printf(" *** Benchmarking %s SPMV_T..\n",sparsekw);
	tic(); for i=1:n; r+=om.'*v; end; tmt=toc();
	TMflops=oppnz*n*2.0*mnz/(10^6 * tmt);
	printf("%s '%s' %d spMVT for %d nnz in  %.4f secs, so %10.2f Mflops\n",mn,sparsekw,n,mnz,tmt, TMflops);

	if(ski<3);
		csvlstring=sprintf("%s%s",csvlstring," n at amflops umt UMflops tmt TMflops");
		csvdstring=sprintf("%s%s%d%s%f%s%f%s%f%s%f%s%f%s%f",csvdstring,sep,n,sep,at,sep,amflops,sep,umt,sep,UMflops,sep,tmt,sep,TMflops);
	endif
	disp " "
end 
	++f;
	# Uncomment following lines for benchmark-oriented output:
	#printf("%s\n",csvlstring);
	#printf("%s\n",csvdstring);
end 

disp " ***********************************************************************"
endif	# nargin > 0
