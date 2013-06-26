function mbench(fname)
addpath ext/
mp=matlabpath();
n=10;
for f=1:nargin
	MB=1024*1024;
	mmn=fname';
	mn=strtrim(mmn');
	tic();
	matlabpath('ext');
	mn;
	nm=mmreadm(mn);
	matlabpath(mp);
	%fsz=stat(mn).size;
	rnz=nnz(nm);
	rt=toc();
	[ia,ja,va]=find(nm);
	%printf('%s: %.2f MBytes read in  %.4f s (%10.2f MB/s)\n',mn',fsz/MB,rt,fsz/(rt*MB));
	%ia=ia'; ja=ja'; va=va';
for ski=1:1
	% FIXME: what about symmetry ?
	sparsekw='sparse';
	if(ski==2)sparsekw='sparsersb';end
	tic();
	for i=1:n;  om=sparse(ia,ja,va); end
	at=toc();
	mnz=nnz(om);
	amflops=n*2.0*mnz/(10^6 * at);
	%printf('%s (%s) %d spBLD for %d nnz in  %.4f secs, so %10.2f Mflops\n',mn',sparsekw,n,rnz,at,amflops);
	amflops
	%rm=sparsersb(ia,ja,va);% UNFINISHED
	v=linspace(1,1,size(om,1))';
	r=v;
	tic(); for i=1:n r=r+om*v; end ; umt=toc();
	UMflops=n*2.0*mnz/(10^6 * umt);
	UMflops
	%printf('%s (%s) %d spMV  for %d nnz in  %.4f secs, so %10.2f Mflops\n',mn',sparsekw,n,mnz,umt, UMflops);
	tic(); for i=1:n r=r+om.'*v; end ; tmt=toc();
	TMflops=n*2.0*mnz/(10^6 * tmt);
	TMflops
	%printf('%s (%s) %d spMV  for %d nnz in  %.4f secs, so %10.2f Mflops\n',mn',sparsekw,n,mnz,tmt, TMflops);
end 
end 
%printf('benchmark terminated successfully\n');
quit
end
