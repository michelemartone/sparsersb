#!/usr/bin/octave -q
#
# A comparative tester for sparsersb.
#
# TODO:
#
# - shall integrate with the rsb.m tester
# - isequal(find(a),find(b)) only checks for pattern!
# - isequal(sparsersb(..),sparsersb(..)) is unfinished !
# - need NZMAX as last arg testing
# - in sparsersb, the == operator is not yet handled natively
# - need testing for find(M,<something>?)
# - seems like non-square matrices are not tested
# - shall test +-, &=, --, ++, .', ./, .\, /, -, +, .*, ./, .^, +0, ==, <=, >=, >, <, |, &
# 

function ase=are_spm_equal(OM,XM)
	if(nnz(XM)!=nnz(OM));ase=0; return; end
	if(columns(XM)!=columns(OM));ase=0; return; end
	if(rows(XM)!=rows(OM));ase=0; return; end
	if(length(XM)!=length(OM));ase=0; return; end
	if(size(XM)!=size(OM));ase=0; return; end
	if(full(XM)!=full(OM));ase=0; return; end
	if((3*XM)!=(3*OM));ase=0; return; end
	if((XM/2)!=(OM/2));ase=0; return; end
	#if((XM.^2)!=(OM.^2));ase=0; return; end
	if((-XM)!=(-OM));ase=0; return; end
	for ri=1:rows(XM)
		if(XM(ri,:)!=OM(ri,:));ase=0; return; end
	end
	for ci=1:columns(XM)
		if(XM(:,ci)!=OM(:,ci));ase=0; return; end
	end
	for ri=1:rows(XM)
	for ci=1:columns(XM)
		if(XM(ri,ci)!=OM(ri,ci));ase=0; return; end
	end
	end
	ase=1;
	if(XM(:,:)!=OM(:,:));ase=0; return; end
	[oi,oj,ov]=find(OM);
	[xi,xj,xv]=find(XM);
	ase&=isequal(oi,xi);
	ase&=isequal(oj,xj);
	ase&=isequal(ov,xv);
end

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

function match=testsprsb(OM,XM)
	match=1;
	# FIXME: shall see in detail whether there are not too many conversions here..
	[oi,oj,ov]=find(OM);
	RM=sparsersb(oi,oj,ov);
	match&=isequal(find(RM),find(OM));
	match&=isequal(find(RM),find(XM));
	clear RM;
	RM=sparsersb(oi,oj,ov,size(OM)(1),size(OM)(2));
	match&=isequal(find(RM),find(OM));
	match&=isequal(find(RM),find(XM));
	clear RM;
	RM=sparsersb(full(OM));
	match&=isequal(find(RM),find(OM));
	match&=isequal(find(RM),find(XM));
	clear RM;
	RM=sparsersb([oi;1;1],[oj;1;1],[ov;-1;1]);
	match&=isequal(find(RM),find(OM));
	match&=isequal(find(RM),find(XM));
	clear RM;
	nr=max(oi);
	nc=max(oj);
	RM=sparsersb([oi;1;1],[oj;1;1],[ov;-1;1],nr,nc,"sum")
	match&=are_spm_equal(RM,OM);
	match&=are_spm_equal(RM,XM);
	clear RM;
	RM=sparsersb([oi;1;1],[oj;1;1],[ov;-2;1],nr,nc,"sum");
	match&=!are_spm_equal(RM,OM);
	match&=!are_spm_equal(RM,XM);
	clear RM;
	RM=sparsersb([oi;nr+1;nr+1],[oj;nc+1;nc+1],[ov;-1;1],nr+1,nc+1,"unique");
	match&=!are_spm_equal(RM,OM);
	match&=!are_spm_equal(RM,XM);
	clear RM;
	testmsg(match,"constructors");
end

function match=testfind(OM,XM)
	match=1;
	match&=isequal(find(OM),find(XM));
	match&=isequal(([oi,oj]=find(OM)),([xi,xj]=find(XM)));
	match&=isequal(([oi,oj,ov]=find(OM)),([xi,xj,xv]=find(XM)));
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
	#sparse(spdiag(OM))
	#sparse(spdiag(XM))
	#match=(sparse(spdiag(OM))==sparse(spdiag(XM)))
	#OM,XM
	#diag(OM)
	#diag(XM)
	match=1;
	if(diag(OM)==diag(XM));match=1;else match=0;end
	#match=(diag(OM)==diag(XM)); # TODO: understand why the following syntax is problematic !
	#match=(spdiag(OM)==spdiag(XM));
	testmsg(match,"diagonal");
end

function match=testpcgm(OM,XM)
	# FIXME! This test ignores OM and XM !
	match=1;
	A=sparse   ([11,12;21,23]);X=[11;22];B=A*X;X=[0;0];
	[OX, OFLAG, ORELRES, OITER, ORESVEC, OEIGEST]=pcg(A,B);
	A=sparsersb([11,12;21,23]);X=[11;22];B=A*X;X=[0;0];
	[XX, XFLAG, XRELRES, XITER, XRESVEC, XEIGEST]=pcg(A,B);
	match&=(OFLAG==XFLAG);# FIXME: a very loose check!
	match&=(OITER==XITER);# FIXME: a very loose check!
	#
	# http://www.gnu.org/software/octave/doc/interpreter/Iterative-Techniques.html#Iterative-Techniques
	#n = 10;
	n = 10+size(XM)(1,1)
	clear A OX XX;
	A = diag (sparse (1:n));
	#A = A + A';
	b = rand (n, 1);
	[l, u, p, q] = luinc (A, 1.e-3);
	[OX, OFLAG, ORELRES, OITER, ORESVEC, OEIGEST]= pcg (          A ,b);
	[XX, XFLAG, XRELRES, XITER, XRESVEC, XEIGEST]= pcg (sparsersb(A),b);
	match&=(sum(OX-XX)==0.0);# FIXME: a very brittle check!
	#
	#function y = apply_a (x)
	#	y = [1:N]' .* x;
	#endfunction
	[OX, OFLAG, ORELRES, OITER, ORESVEC, OEIGEST]= pcg (          A ,b, 1.e-6, 500, l,u);
	[XX, XFLAG, XRELRES, XITER, XRESVEC, XEIGEST]= pcg (sparsersb(A),b, 1.e-6, 500, l,u);
	match&=(sum(OX-XX)==0.0);# FIXME: a very brittle check!
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
	match&=are_spm_equal(OX,XX);
	OX=OM.'*B; XX=XM.'*B;
	match&=isequal(OX,XX);# FIXME: a very loose check!
	match&=are_spm_equal(OX,XX);
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

function match=testnorm(OM,XM)
	match=1;
	if(isreal(OM))
		match&=isequal(full(normest(OM)),full(normest(XM)));
	end
	testmsg(match,"norms");
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

function match=tests(OM,XM,M)
	if(nargin>2)
		M
	endif
	match=1;
	if nnz(OM)>0
	match&=testsprsb(OM,XM);
	end
	match&=testinfo(OM,XM);
	match&=testdims(OM,XM);
	match&=testdiag(OM,XM);
	match&=testfind(OM,XM);
	match&=testelms(OM,XM);
	match&=testasgn(OM,XM);
	if nnz(OM)>1
		match&=testpcgm(OM,XM);
		match&=testmult(OM,XM);
		match&=testspsv(OM,XM);
		match&=testnorm(OM,XM);
	end
	match&=testscal(OM,XM);
	match&=testadds(OM,XM);
	testmsg(match,"overall (for this matrix)");
end

match=1;
mtn=1;

if (strchr(sparsersb("?"),"Z")>0)
	mtn++;
endif

for mti=1:mtn
wc=(mti==2);

dim=3;
#M=(rand(dim)>.8)*rand(dim);M(1,1)=11;

M=[0];
OM=sparse(M); XM=sparsersb(M);
match&=tests(OM,XM);

for k=1:6
M=[eye(k)];
if(wc)M+=M*i;end
OM=sparse(M); XM=sparsersb(M);
match&=tests(OM,XM,M);
end

#M=zeros(4)+sparse([1,2,3,2,4],[1,2,3,1,4],[11,22,33,21,44]);
#if(wc)M+=M*i;end
#OM=sparse(M); XM=sparsersb(M);
#match&=tests(OM,XM);

for k=3:6
M=zeros(k)+sparse([linspace(1,k,k),2],[linspace(1,k,k),1],[11*linspace(1,k,k),21]);
if(wc)M+=M*i;end
OM=sparse(M); XM=sparsersb(M);
match&=tests(OM,XM);
end

#M=tril(ones(10))+100*diag(10);
#OM=sparse(M); XM=sparsersb(M);
#match&=tests(OM,XM);

#M=hilb(10)+100*diag(10);
#OM=sparse(M); XM=sparsersb(M);
#match&=tests(OM,XM);

M=diag(10);
if(wc)M+=M*i;end
OM=sparse(M); XM=sparsersb(M);
match&=tests(OM,XM);

#M=diag(10)+sparse([1,10],[10,10],[.1,1]);
#OM=sparse(M); XM=sparsersb(M);
#match&=tests(OM,XM);

end

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
