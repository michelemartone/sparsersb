#!/usr/bin/octave -q

function rsbb_sum(A,B)
	tic
	C=A+B;
	toc
end

function rsbb_mul(A,B)
	tic
	C=A*B;
	toc
end

function rsbb_spmm(A,nrhs)
	X=ones(size(A,2),nrhs);
	T=100;
	tic
	for t=1,T;
		Y=A*X;
	end
	toc
end

function rsbb_scale(A)
	T=100;
	alpha=1.1;
	tic
	for t=1,T;
	A.*=alpha;
	end
	toc
end

function rsbb_spmv(A)
	rsbb_spmm(A,1)
end

function rsbb_diag(A)
	tic
	d=diag(A);
	toc
end

#n=2000;
#n=200;
n=800;
th=.4;
oA=sparse(rand(n)>th);
oB=sparse(rand(n)>th);
rA=sparsersb(oA);
rB=sparsersb(oB);

rsbb_spmv(oA)
rsbb_spmv(rA)

rsbb_spmm(oA,10)
rsbb_spmm(rA,10)

rsbb_mul(oA,oB)
rsbb_mul(rA,rB)

rsbb_sum(oA,oB)
rsbb_sum(rA,rB)

rsbb_diag(oA)
rsbb_diag(rA)

rsbb_scale(oA)
rsbb_scale(rA)


