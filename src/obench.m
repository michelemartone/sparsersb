# once complete, this program will benchmark our matrix implementation against octave's

#for n_=1:6*0+1
for n_=1:6
	n=n_*1000;
	m=k=n;
	# making vectors
	b=linspace(1,1,n)';
	ox=linspace(1,1,n)';
	bx=linspace(1,1,n)';
	# making matrices
	r=(rand(n)>.6);
	om=sparse(r);
	#bm=sparsevbr(om);
	bm=sparsersb(sparse(om));
	#bm=sparsersb3(sparse(om));
	# stats
	nz=nnz(om);
	flops=2*nz;
	M=10^6;
	## spmv
	ot=-time; ox=om*b; ot+=time;
	#
	bt=-time; bx=bm*b; bt+=time;
	t=ot; p=["octave-",version]; mflops=(flops/M)/t;
	printf("%s\t%d\t%d\t%d\t%g\t%s\n","*",m,k,nz,mflops, p);
	t=bt; p=["RSB"]; mflops=(flops/M)/t;
	printf("%s\t%d\t%d\t%d\t%g\t%s\n","*",m,k,nz,mflops, p);

	## spmvt
	ot=-time; ox=om.'*b; ot+=time;
	#
	bt=-time; bx=bm.'*b; bt+=time;
	t=ot; p=["octave-",version]; mflops=(flops/M)/t;
	printf("%s\t%d\t%d\t%d\t%g\t%s\n","*",m,k,nz,mflops, p);
	t=bt; p=["RSB"]; mflops=(flops/M)/t;
	printf("%s\t%d\t%d\t%d\t%g\t%s\n","*",m,k,nz,mflops, p);

	## spgemm
	ot=-time; ox=om*om; ot+=time;
	#
	bt=-time; bx=bm*bm; bt+=time;
	t=ot; p=["octave-",version]; mflops=n*(flops/M)/t;
	printf("%s\t%d\t%d\t%d\t%g\t%s\n","OCT_SPGEMM",m,k,nz,mflops, p);
	t=bt; p=["RSB"]; mflops=n*(flops/M)/t;
	printf("%s\t%d\t%d\t%d\t%g\t%s\n","RSB_SPGEMM",m,k,nz,mflops, p);
endfor
