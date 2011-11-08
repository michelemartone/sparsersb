sz=2;
#_rm=sparsersb(sparse(1)) # debug this
#_rm=sparsersb((rand(sz)))
#_rm=sparsersb([1,2,2],[1,1,2],[11,21,22])
#_rm=sparsersb([11,12;21,22]); exit
_rm=sparsersb([1,1,1,2,2],[1,1,1,1,2],[1,2,8,21,22],"summation")
#exit
#_rm=sparsersb([1,1,1,2,2],[1,1,1,1,2],[1,2,8,21,22],"unique")
_om=sparse([1,2,2],[1,1,2],[11,21,22])
#_rm=sparsersb([1,1,2],[1,2,2],[11,12,22])
#_om=sparse   ([1,1,2],[1,2,2],[11,12,22])
#_rm=sparsersb([1,1,2,2],[1,2,1,2],[11,12,21,22])
#_om=sparse([1,1,2,2],[1,2,1,2],[11,12,21,22])
#_rm=sparsersb(sparse(rand(sz)))
_dm=rand(sz)
_rm2=_rm
_om2=_rm
rama=(rand(sz))
_rm2=sparsersb(rama)
_om2=sparse(rama)
_rm2=[]
_om2=[]
#exit
# uhm..
_rm=transpose(_rm)
_om=transpose(_om)
_rm=transpose(_rm)
_om=transpose(_om)
#exit

#_rm

#exit
nnz(_rm)== nnz(_om)
columns(_rm)== columns(_om)
rows(_rm)== rows(_om)
length(_rm)==length(_om)
size(_rm)== size(_om)
#exit

# the following is a problem (FIXME)
#em=sparsersb([]);

if false
# FIXME: should finish implementing subsref ! This branch tests unfinished features! 
_rm(1,1)
_om(1,1)
_rm(1,1)==_om(1,1)
#exit
#_rm(1,1)=1

_rm(1,:)
_om(1,:)
# FIXME
#_rm(1,:)=_om(1,:)
#_rm(1,:)=1
#exit

_om(:,1)
_rm(:,1)
#_om(:,1)==_rm(:,1)
#_rm(:,1)=1
#exit

_rm(:,:)
_om(:,:)
#_rm(:,:)==_om(:,:)
#_rm(:,:)=1
#exit
_rm([1,1],[2,2])
#_rm([1,1],[2,2])=1

# and dotref ?
# we have an empty stub to complete, here 
_rm(1,1)=1
_rm([1:2],[1:2])=1
else
	# These are the only allowed operations so far:
	_rm
	#_rm(1,1)
	#_om(1,1)
	#_rm(1,1)==_om(1,1)
	#date,exit
%_rm(1,1)=1
endif

#_rm+=1

#_rm*=1

#_rm++

#_rm--

#_rm(:,:)=1

#__rm=sparsersb(1)
__rm=_rm

#sparsersb(_rm)'

-_rm
x=ones(sz,1)
#_rm==_om # FIXME: TODO
#_rm
#_om # FIXME: TODO
y=_rm*x
y=_om*x
_rm*x==_om*x
y=_rm.'*x
y=_om.'*x
#y=transpose(_rm)*x
#y=_rm*(x')
#exit
#y=(_rm')*x
#s=_om./x
#s=_rm./x
#s=_om.\x
#s=_rm.\x
#s=_om/x
#s=_rm/x

#_om=_rm
#x
#xx=x
s=_rm\x
#x
z=_om\x
#x
help sparsersb
_rm+_rm
_om+_om
_rm-_rm
_om-_om
_rm*_rm
_om*_om
_rm.*2
_rm./2
_rm.^2
_rm-100
_rm+100
_rm+0
_rm==_dm
_rm<=_dm
_rm<_dm
_rm>=_dm
_rm>_dm
_rm|_dm
_rm&_dm
#_rm--
#_rm++
#_rm=_dm
exit
(_rm*x)
u:srm*(y)
_rm*(_rm*x)

#_rm==_rm

##
exit

###############################################################


if 1
	n=4000
#	n=2
	r=rand(n)>.7;
	printf("matrix ok\n");
	#r*=2; # a matrix of 2's (STRANGE : uncommenting this make spmv faster :) )
	r*=1; # the way to squeeze octave of a two-some factor or so (probably it gets elements reordered..)
else
	r=[1,2,3;4,5,6,;7,8,9];
	#r=mmread("/usr/local/matrices/raefsky4.mtx");
	#r=mmread("/usr/local/matrices/bayer02.mtx");
	n=size(r,1);
end
for i=1:n
	r(i,i)=1;
end
r;
nnz_=nnz(r)

o=linspace(1,1,n)';

## rsb
#printf("making rsb from dense\n");
#rm=sparsersb(r);
printf("making rsb from sparse\n");
rm=sparsersb(sparse(r));
printf("making spmv\n");
vr=rm*o;
#printf("  rsb    matmult : %g\n",time-t);

## octave native
sm=sparse(r);

# TODO : use 'tic' and 'toc'

for i=1:3
t=time;
sr=sm*o;
t1=time-t;
printf("  octave matmult : %g\n",t1);
endfor

printf("sr vr sr-vr\n");

[sr';vr';(sr-vr)';]';

if nnz(sr-vr)
  printf("BAD! %d nonzeros don't match!\n",nnz(sr-vr));
endif
#sr-vr
#sm-rm

