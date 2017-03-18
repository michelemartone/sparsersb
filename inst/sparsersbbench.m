#!/usr/bin/octave -q
# 
#  Copyright (C) 2011-2016   Michele Martone   <michelemartone _AT_ users.sourceforge.net>
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
# TODO: document this file's functions so they get installed and are properly usable.
# TODO: sprand should not be used in a consistent way

1; # This is a script.
if length(getenv("SPARSERSB_TEST")) == 0 ; pkg load sparsersb ; end

function dt=sparsersbbench__(precmd,cmd,postcmd,mint)
# ..
	eval(precmd);
	nops=0;
	tic();
	do
		++nops;
		eval(cmd);
	until ((dt=toc())>=mint)
	dt/=nops;
	eval(postcmd);
end 

function speedup=sparsersbbench_(gprecmd,precmd,cmd,postcmd,gpostcmd,mint)
# ...
	rprecmd=strrep(precmd,"sparsersb","sparse");
	rcmd=strrep(cmd,"sparsersb","sparse");
	rpostcmd=strrep(postcmd,"sparsersb","sparse");
	dots=";";
	once=[precmd,dots,cmd,dots,postcmd];
	#eval(once); printf("iterating %s\n",once);
#	dots="...";
	predots=sprintf(";tic;do;");
	postdots=sprintf(";until ((dt=toc())>=%f);",mint);
	all=[gprecmd,dots,precmd,predots,cmd,postdots,postcmd,dots,gpostcmd];
#	printf("will see speedup for %s\n",all);
	printf("#%s #-> speedup is...",all);
	dtr=sparsersbbench__([gprecmd,"", precmd,""], cmd,[ postcmd,":",gpostcmd],mint);
	dto=sparsersbbench__([gprecmd,"",rprecmd,""],rcmd,[rpostcmd,":",gpostcmd],mint);
	speedup=dto/dtr;
	printf("%.2f\n",speedup);
#	printf("%.2f speedup for %s\n",speedup,all);
end

function sparsersbbench_battery(mstring,mint)
	rinitstr=["A=sparsersb(",mstring,");nr=size(A)(1);nc=size(A)(2);"];
	finitstr=["A=full(",mstring,");"];
	cinitstr=["M=sparse(",mstring,");[ia,ja,va]=find(M);nr=size(M)(1);nc=size(M)(2);"];
	sparsersbbench_("",[rinitstr,""],"A.*=2.0;","clear A","",mint);
	sparsersbbench_("",[rinitstr,""],"A.*=2.5;","clear A","",mint);
	sparsersbbench_("",[rinitstr,""],"A./=2.0;","clear A","",mint);
	sparsersbbench_("",[rinitstr,""],"A./=2.5;","clear A","",mint);
	#sparsersbbench_("",[rinitstr,""],"A.*=0.0;","clear A","",mint);
	#sparsersbbench_("",[rinitstr,""],"A./=0.0;","clear A","",mint);
	sparsersbbench_("",[rinitstr,""],"A.^=2.5;","clear A","",mint);
	sparsersbbench_("",[rinitstr,""],"A.^=2.0;","clear A","",mint);
	sparsersbbench_("",[rinitstr,""],"A.^=0.5;","clear A","",mint);
	sparsersbbench_("",[cinitstr,""],"C=sparsersb(ia,ja,va,nr,nc);clear C;","clear A C ia ja va","",mint);
	sparsersbbench_("",[cinitstr,""],"C=sparsersb(ia,ja,va);clear C;","clear A C ia ja va","",mint);
	sparsersbbench_("",[cinitstr,""],"C=sparsersb(ja,ia,va);clear C;","clear A C ia ja va","",mint);
	sparsersbbench_("",[finitstr,""],"C=sparsersb(A);clear C;","clear A C","",mint);
	sparsersbbench_("",[rinitstr,""],"C=A;          ;clear C;","clear A C","",mint);
	sparsersbbench_("",[rinitstr,""],"C=A.';        ;clear C;","clear A C","",mint);
	sparsersbbench_("",[rinitstr,""],"C=transpose(A);clear C;","clear A C","",mint);
	for nrhs=1:3
	nrhss=sprintf("%d",nrhs);
	sparsersbbench_("",[rinitstr,"C=ones(nr,",nrhss,");B=C;"],"C=A*B;","clear A B C","",mint);
	sparsersbbench_("",[rinitstr,"C=ones(nr,",nrhss,");B=C;"],"C=A.'*B;","clear A B C","",mint);
	eval(finitstr);
	if (tril(A)==A) || (triu(A)==A)
	sparsersbbench_("",[rinitstr,"C=ones(nr,",nrhss,");B=C;"],"C=A\\B;","clear A B C","",mint);
	sparsersbbench_("",[rinitstr,"C=ones(nr,",nrhss,");B=C;"],"C=A.'\\B;","clear A B C","",mint);
	end
	end
	clear A;
	sparsersbbench_("",[rinitstr,"D=ones(nr,1);"],"D=diag(A);","clear A D","",mint);
	sparsersbbench_("",[rinitstr,"B=A;"],"C=A+B;","clear A B C","",mint);
	sparsersbbench_("",[rinitstr,"B=A;"],"C=A.'+B;","clear A B C","",mint);
	sparsersbbench_("",[rinitstr,"B=A;"],"C=A*B;clear C","clear A B C","",mint);
	sparsersbbench_("",[rinitstr,"B=A;"],"C=A.'*B;clear C","clear A B C","",mint);
end


btime=1.0;
if false ;
# shall use: [x, flag, prec_res_norm, itcnt]
sparsersbbench_("n=1000; k=15; oA=k*eye(n)+sprandn(n,n,.2); b=ones(n,1); P=diag(diag(oA));","A=sparsersb(oA);","[x, flag] = gmres (A, b, [], 1e-7, n, P);","clear b P","clear oA",btime);
sparsersbbench_("n=2000; k=150; oA=k*eye(n)+sprandn(n,n,.2); b=ones(n,1); P=diag(diag(oA));","A=sparsersb(oA);","[x, flag] = gmres (A, b, [], 1e-7, n, P);","clear b P","clear oA",btime);
sparsersbbench_("n=4000; k=1500; oA=k*eye(n)+sprandn(n,n,.2); b=ones(n,1); P=diag(diag(oA));","A=sparsersb(oA);","[x, flag] = gmres (A, b, [], 1e-7, n, P);","clear b P","clear oA",btime);
sparsersbbench_("n=6000; k=3500; oA=k*eye(n)+sprandn(n,n,.2); b=ones(n,1); P=diag(diag(oA));","A=sparsersb(oA);","[x, flag] = gmres (A, b, [], 1e-7, n, P);","clear b P","clear oA",btime);
end

#for diml=0:0
for diml=1:11
#for diml=11:11
#for diml=3:3
#for cadd=1:1
#for cadd=0:0
for cadd=0:1
	btime=1.0;
	if(diml<7)btime=0.1;end
	dim=2^diml;
	#is=sprintf("ones(%d)",dim);

	cmul=sprintf("(1+i*%d)",cadd);
	is=sprintf("ones(%d).*%s",dim,cmul);
	sparsersbbench_battery(is,btime)
	is=sprintf("tril(ones(%d).*%s)",dim,cmul);
	sparsersbbench_battery(is,btime)
	is=sprintf("diag(ones(%d,1)).*%s",dim,cmul);
	sparsersbbench_battery(is,btime)

       	# FIXME: follow non repeatable experiments :)
	is=sprintf("(diag(ones(%d,1))+sprand(%d,%d,0.1)).*%s",dim,dim,dim,cmul);
	sparsersbbench_battery(is,btime)
	is=sprintf("(diag(ones(%d,1))+sprand(%d,%d,0.4)).*%s",dim,dim,dim,cmul);
	sparsersbbench_battery(is,btime)

	# FIXME: need a non-square matrices testing-benchmarking snippet
end
end

printf "All done."
