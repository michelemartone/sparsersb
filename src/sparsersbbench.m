#!/usr/bin/octave -q
# TODO: document this file's functions so they get installed and are properly usable.

function dt=sparsersbbench(precmd,cmd,postcmd,mint)
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
	dtr=sparsersbbench([gprecmd, precmd], cmd,[ postcmd,gpostcmd],mint);
	dto=sparsersbbench([gprecmd,rprecmd],rcmd,[rpostcmd,gpostcmd],mint);
	speedup=dto/dtr;
	printf("%.2f speedup for %s\n",speedup,all);
end

function sparsersbbench_battery(mstring,mint)
	rinitstr=["A=sparsersb(",mstring,");"];
	finitstr=["A=full(",mstring,");"];
	cinitstr=["[ia,ja,va]=find(sparse(",mstring,"));"];
	sparsersbbench_("",[cinitstr,""],"C=sparsersb(ia,ja,va);clear C;","clear C ia ja va","",mint);
	sparsersbbench_("",[cinitstr,""],"C=sparsersb(ja,ia,va);clear C;","clear C ia ja va","",mint);
	sparsersbbench_("",[finitstr,""],"C=sparsersb(A);clear C;","clear C","",mint);
	sparsersbbench_("",[rinitstr,""],"C=A.';        ;clear C;","clear C","",mint);
	sparsersbbench_("",[rinitstr,""],"C=transpose(A);clear C;","clear C","",mint);
	sparsersbbench_("",[rinitstr,"B=A;"],"C=A*B;clear C","clear B C","",mint);
	sparsersbbench_("",[rinitstr,"B=A;"],"C=A.'*B;clear C","clear B C","",mint);
	sparsersbbench_("",[rinitstr,"D=ones(size(A)(1),1);"],"D=diag(A);","clear D","",mint);
	for nrhs=1:3
	nrhss=sprintf("%d",nrhs);
	sparsersbbench_("",[rinitstr,"C=ones(size(A)(1),",nrhss,");B=C;"],"C=A*B;","clear B C","",mint);
	sparsersbbench_("",[rinitstr,"C=ones(size(A)(1),",nrhss,");B=C;"],"C=A.'*B;","clear B C","",mint);
	eval(finitstr);
	if (tril(A)==A) || (triu(A)==A)
	sparsersbbench_("",[rinitstr,"C=ones(size(A)(1),",nrhss,");B=C;"],"C=A\\B;","clear B C","",mint);
	sparsersbbench_("",[rinitstr,"C=ones(size(A)(1),",nrhss,");B=C;"],"C=A.'\\B;","clear B C","",mint);
	end
	end
	clear A;
	sparsersbbench_("",[rinitstr,""],"A.*=2.0;","","",mint);
	sparsersbbench_("",[rinitstr,""],"A./=2.0;","","",mint);
	#sparsersbbench_("",[rinitstr,""],"A.*=0.0;","","",mint);
	#sparsersbbench_("",[rinitstr,""],"A./=0.0;","","",mint);
	sparsersbbench_("",[rinitstr,""],"A.^=2.0;","","",mint);
	sparsersbbench_("",[rinitstr,""],"A.^=0.5;","","",mint);
	sparsersbbench_("",[rinitstr,"B=A;"],"C=A+B;","clear B C","",mint);
	sparsersbbench_("",[rinitstr,"B=A;"],"C=A.'+B;","clear B C","",mint);
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
#for diml=1:11
for diml=1:11
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
end
end

printf "All done."
