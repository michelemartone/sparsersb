#!/usr/bin/octave -q
# a benchmark program for octave/matlab
# TODO: fix output format 
# TODO: correct symmetric / hermitian matrices handling
# TODO: sound, time-and-runs-based benchmarking criteria 

n=10;

function printbenchline(matrixname,opname,sw,times,nnz,tottime,mxxops,bpnz,msstr)
	printf("FIXME (temporary format)\n");
	printf("%s %s %s %d %d %.4f %10.2f %.4f %s\n",matrixname,opname,sw,times,nnz,tottime,mxxops,bpnz,msstr);
end

if nargin <= 0
# DGEMV benchmark
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
quit
endif

# if nargin > 0, we continue


source("ext/mminfo.m");
source("ext/mmread.m");
source("ext/mmwrite.m");

#matrices=ls("*.mtx")';
f=1;
uc=2;
while f<=nargin
	MB=1024*1024;
	mmn=cell2mat(argv()(f))';
	mn=strtrim(mmn');
	tic();
	#nm=mmread(mn);
	[nm,nrows,ncols,entries,rep,field,symm]=mmread(mn);
	#if(symm=="symmetric")uc+=2;endif
	if(strcmp(symm,"symmetric"))uc+=1;endif
	fsz=stat(mn).size;
	rt=toc();
	[ia,ja,va]=find(nm);
	printf("%s: %.2f MBytes read by mmread in  %.4f s (%10.2f MB/s)\n",mn',fsz/MB,rt,fsz/(rt*MB));
	#ia=ia'; ja=ja'; va=va';
for ski=1:uc
	oppnz=1;
	# FIXME: what about symmetry ?
	sparsekw="sparse";
	if(ski==2)sparsekw="sparsersb";endif
	if(ski==3);
		oppnz=2;
		sparsekw="sparsersb";
		tic(); [nm]=sparsersb(mn); rt=toc();
		sparsersb(nm,"info")
		printf("%s: %.2f MBytes read by librsb in  %.4f s (%10.2f MB/s)\n",mn',fsz/MB,rt,fsz/(rt*MB));
	endif
	if(ski==4);
		nm=tril(nm);
	endif
	[ia,ja,va]=find(nm);
	rnz=nnz(nm);
	printf("benchmarking %s\n",sparsekw);
	#printf("symmetry ? %s\n",issymmetric(sparse(nm)));
	mrc=rows(nm); mcc=columns(nm);


	if(ski!=3);
	tic();
	eval(["for i=1:n;  om=",sparsekw,"(ia,ja,va,mrc,mcc,\"summation\"); end"]);
	printf("benchmarking %s\n",sparsekw);
	at=toc();
	mnz=nnz(om);
	amflops=n*2.0*mnz/(10^6 * at);
	printf("%s (%s) %d spBLD for %d nnz in  %.4f secs, so %10.2f Mflops\n",mn',sparsekw,n,rnz,at,amflops);
	else
	mnz=rnz;
	end

	#rm=sparsersb(ia,ja,va);# UNFINISHED
	r=v=linspace(1,1,size(om,1))';
	tic(); for i=1:n; r+=om  *v; end; umt=toc();
	UMflops=oppnz*n*2.0*mnz/(10^6 * umt);
	printf("%s (%s) %d spMV  for %d nnz in  %.4f secs, so %10.2f Mflops\n",mn',sparsekw,n,mnz,umt, UMflops);
	bpnz=-1;  # FIXME: bytes per nonzero!
	msstr="?";# FIXME: matrix structure string!
	# FIXME: finish the following!
	#printbenchline(mn',"spMV",sparsekw,n,mnz,umt, UMflops,bpnz,msstr);
	#
	tic(); for i=1:n; r+=om.'*v; end; tmt=toc();
	TMflops=oppnz*n*2.0*mnz/(10^6 * tmt);
	printf("%s (%s) %d spMVT for %d nnz in  %.4f secs, so %10.2f Mflops\n",mn',sparsekw,n,mnz,tmt, TMflops);
end 
	++f;
end 

printf("benchmark terminated\n");

