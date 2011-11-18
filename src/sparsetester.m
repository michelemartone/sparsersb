#!/usr/bin/octave -q
#
# A comparative tester for sparsersb.
#

function testmsg(match,tname,erreason)
	if(match)
		printf(" [*] %s test passed",tname)
	else
		printf(" [!] %s test failed",tname)
	end
	if(nargin<3)
		printf(".\n")
	else
		printf(" ().\n",erreason)
	end
end

function match=testinfo(OM,XM)
	printf("will test types \"%s\" and \"%s\"\n",typeinfo(OM),typeinfo(XM))
	match=1;
end

function match=testdims(OM,XM)
	match=1;
	match&=(rows(OM)==rows(XM));
	match&=(columns(OM)==columns(XM));
	match&=(nnz(OM)==nnz(XM));
	testmsg(match,"dimensions");
end

function match=testfind(OM,XM)
	match=1;
	match&=isequal(find(OM),find(XM));
	match&=isequal(([oi,oj]=find(OM)),([xi,xj]=find(XM)));
	match&=isequal(nonzeros(OM),nonzeros(XM));
	testmsg(match,"find");
end

function match=testasgn(OM,XM)
	match=1;
	nr=rows(OM);
	nc=columns(OM);
	for i=1:nr
	for j=1:nc
		#printf("%d %d / %d %d\n", i,j,nr,nc)
		#OM, XM
		#printf("%d %d %d\n", i,j,XM(i,j));
		if(XM(i,j))
			nv=rand(1);
			OM(i,j)=nv;
			XM(i,j)=nv;
 		end
		#OM, XM
		#exit
	endfor
	endfor
	for i=1:nr
	for j=1:nc
		if(OM(i,j))match&=isequal(OM(i,j),XM(i,j)); end;
	endfor
	endfor
	testmsg(match,"asgn");
end

function match=testelms(OM,XM)
	match=1;
	nr=rows(OM);
	nc=columns(OM);
	for i=1:nr
	for j=1:nc
		if(OM(i,j)!=XM(i,j)); match*=0; end
	endfor
	endfor
	testmsg(match,"elems");
end

function match=testdiag(OM,XM)
	match=(diag(OM)==diag(XM));
	testmsg(match,"diagonal");
end

function match=testpcgm(OM,XM)
	# FIXME!
	match=1;
	A=sparse   ([11,12;21,23]);X=[11;22];B=A*X;X=[0;0];
	[OX, OFLAG, ORELRES, OITER, ORESVEC, OEIGEST]=pcg(A,B);
	A=sparsersb([11,12;21,23]);X=[11;22];B=A*X;X=[0;0];
	[XX, XFLAG, XRELRES, XITER, XRESVEC, XEIGEST]=pcg(A,B);
	match&=(OFLAG==XFLAG);# FIXME: a very loose check!
	match&=(OITER==XITER);# FIXME: a very loose check!
	testmsg(match,"pcg");
end

function match=testmult(OM,XM)
	match=1;
	B=ones(rows(OM),1);
	B=[B';2*B']';
	#OM, XM
	OX=OM*B; XX=XM*B;
	match&=isequal(OX,XX);# FIXME: a very loose check!
	OX=OM'*B; XX=XM'*B;
	match&=isequal(OX,XX);# FIXME: a very loose check!
	OX=transpose(OM)*B; XX=transpose(XM)*B;
	match&=isequal(OX,XX);# FIXME: a very loose check!
	OX=OM.'*B; XX=XM.'*B;
	match&=isequal(OX,XX);# FIXME: a very loose check!
	testmsg(match,"multiply");
end

function match=testspsv(OM,XM)
	match=1;
	#B=ones(rows(OM),1);
	X=(1:rows(OM))';
	X=[X';2*X']';
	B=OM*X;
	#OM, XM
	OX=OM\B;
       	#B
	XX=XM\B;
	#B
	match&=isequal(OX,XX);# FIXME: a very loose check!
	testmsg(match,"triangular solve");
end

function match=testscal(OM,XM)
	match=1;
	OB=OM;
	XB=XM;
	#OB
	#XB
	OM=OM/2; XM=XM/2;
	match&=isequal(find(OM),find(XM));
	OM=OM*2; XM=XM*2;
	match&=isequal(find(OM),find(XM));
	#
	OM/=2; XM/=2;
	match&=isequal(find(OM),find(XM));
	OM*=2; XM*=2;
	match&=isequal(find(OM),find(XM));
	#
	OM=OM./2; XM=XM./2;
	match&=isequal(find(OM),find(XM));
	OM=OM.*2; XM=XM.*2;
	match&=isequal(find(OM),find(XM));
	#
	OM./=2; XM./=2;
	match&=isequal(find(OM),find(XM));
	OM.*=2; XM.*=2;
	match&=isequal(find(OM),find(XM));
	#
	#
	OM=OM.^2; XM=XM.^2;
	match&=isequal(find(OM),find(XM));
	# FIXME
	OM=OM^2; XM=XM^2;
	match&=isequal(find(OM),find(XM));
	#
	OM=OM.^(1/2); XM=XM.^(1/2);
	match&=isequal(find(OM),find(XM));
	# FIXME
	OM=OM^(1/2); XM=XM^(1/2);
	match&=isequal(find(OM),find(XM));
	#
	match&=isequal(find(OM),find(OB));
	match&=isequal(find(XM),find(XB));
	testmsg(match,"scale");
	OM=OB; XM=XB;
end

function match=testadds(OM,XM)
	match=1;
	OB=OM;
	XB=XM;
	#OB
	#XB
	#
	#
	#i=1;j=1;
	for i=1:rows(OM)
	for j=1:columns(OM)
	if(OM(i,j))
		OM(i,j)+=2; XM(i,j)+=2;
		#OM(2,3)+=2; XM(2,3)+=2;
		match&=isequal(find(OM),find(XM));
		#exit
		OM(i,j)-=2; XM(i,j)-=2;
		match&=isequal(find(OM),find(XM));
	else
		# FIXME: will only work on EXISTING pattern elements, on sparsersb 
		# FIXME: should write a test case for the pattern-changing operation
	end
	endfor
	endfor
	#
	match&=isequal(find(OM),find(OB));
	match&=isequal(find(XM),find(XB));
	testmsg(match,"add and subtract");
	OM=OB; XM=XB;
end

function match=tests(OM,XM)
	match=1;
	match&=testinfo(OM,XM);
	match&=testdims(OM,XM);
	match&=testdiag(OM,XM);
	match&=testfind(OM,XM);
	match&=testelms(OM,XM);
	match&=testasgn(OM,XM);
	match&=testpcgm(OM,XM);
	match&=testmult(OM,XM);
	match&=testspsv(OM,XM);
	match&=testscal(OM,XM);
	match&=testadds(OM,XM);
	testmsg(match,"overall (for this matrix)");
end

match=1;
dim=3;
#M=(rand(dim)>.8)*rand(dim);M(1,1)=11;

#M=[0];
#OM=sparse(M); XM=sparsersb(M);
#match&=tests(OM,XM);

M=[1];
OM=sparse(M); XM=sparsersb(M);
match&=tests(OM,XM);

M=zeros(4)+sparse([1,2,3,2,4],[1,2,3,1,4],[11,22,33,21,44]);
OM=sparse(M); XM=sparsersb(M);
match&=tests(OM,XM);

#M=tril(ones(10))+100*diag(10);
#OM=sparse(M); XM=sparsersb(M);
#match&=tests(OM,XM);

#M=hilb(10)+100*diag(10);
#OM=sparse(M); XM=sparsersb(M);
#match&=tests(OM,XM);

M=diag(10);
OM=sparse(M); XM=sparsersb(M);
match&=tests(OM,XM);

#M=diag(10)+sparse([1,10],[10,10],[.1,1]);
#OM=sparse(M); XM=sparsersb(M);
#match&=tests(OM,XM);

if(match) printf("All tests passed.\n"); else printf("Failure while performing tests!\n");end

# FIXME: shall print a report in case of failure.

#M=zeros(3)+sparse([1,2,3],[1,2,3],[11,22,33]);
#M=sparse([1,2,3],[1,2,3],[11,22,33]);
#XM=sparsepsb(M);
#
#
exit
#
XM
find(XM)
[i,j]=find(XM)
#
exit
#
