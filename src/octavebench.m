#!/usr/bin/octave -q
# a benchmark program for octave/matlab
n=10;

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
while f<=nargin
	MB=1024*1024;
	mmn=cell2mat(argv()(f))';
	mn=strtrim(mmn');
	tic();
	nm=mmread(mn);
	fsz=stat(mn).size;
	rnz=nnz(nm);
	rt=toc();
	[ia,ja,va]=find(nm);
	printf("%s: %.2f MBytes read in  %.4f s (%10.2f MB/s)\n",mn',fsz/MB,rt,fsz/(rt*MB));
	#ia=ia'; ja=ja'; va=va';
for ski=1:2
	# FIXME: what about symmetry ?
	sparsekw="sparse";
	if(ski==2)sparsekw="sparsersb";endif
	mrc=rows(nm); mcc=columns(nm);
	tic();
	eval(["for i=1:n;  om=",sparsekw,"(ia,ja,va,mrc,mcc,\"summation\"); end"]);
	at=toc();
	mnz=nnz(om);
	amflops=n*2.0*mnz/(10^6 * at);
	printf("%s (%s) %d spBLD for %d nnz in  %.4f secs, so %10.2f Mflops\n",mn',sparsekw,n,rnz,at,amflops);
	#rm=sparsersb(ia,ja,va);# UNFINISHED
	r=v=linspace(1,1,size(om,1))';
	tic(); for i=1:n; r+=om  *v; end; umt=toc();
	UMflops=n*2.0*mnz/(10^6 * umt);
	printf("%s (%s) %d spMV  for %d nnz in  %.4f secs, so %10.2f Mflops\n",mn',sparsekw,n,mnz,umt, UMflops);
	tic(); for i=1:n; r+=om.'*v; end; tmt=toc();
	TMflops=n*2.0*mnz/(10^6 * tmt);
	printf("%s (%s) %d spMVT for %d nnz in  %.4f secs, so %10.2f Mflops\n",mn',sparsekw,n,mnz,tmt, TMflops);
end 
	++f;
end 

printf("benchmark terminated\n");

