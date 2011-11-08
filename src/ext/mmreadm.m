% 20111022
% Slightly modified by Michele Martone for Matlab at IPP MPG.
% 20070301
% Slightly modified by Michele Martone for handling complex dense arrays with parenthesis,too
% But there still a bug (original!): the first line MUST be space terminated.. 
function  [A,rows,cols,entries,rep,field,symm] = mmreadm(filename)
%
% function  [A] = mmreadm(filename)
%
% function  [A,rows,cols,entries,rep,field,symm] = mmreadm(filename)
%
%      Reads the contents of the Matrix Market file 'filename'
%      into the matrix 'A'.  'A' will be either sparse or full,
%      depending on the Matrix Market format indicated by
%      'coordinate' (coordinate sparse storage), or
%      'array' (dense array storage).  The data will be duplicated
%      as appropriate if symmetry is indicated in the header.
%
%      Optionally, size information about the matrix can be 
%      obtained by using the return values rows, cols, and
%      entries, where entries is the number of nonzero entries
%      in the final matrix. Type information can also be retrieved
%      using the optional return values rep (representation), field,
%      and symm (symmetry).
%

mmfile = fopen(filename,'rb');%'b' added by MM
if ( mmfile == -1 )
 disp(filename);
 error('File not found');
end;

header = fgets(mmfile);
if (header == -1 )
  error('Empty file.')
end

% NOTE: If using a version of Matlab for which strtok is not
%       defined, substitute 'gettok' for 'strtok' in the 
%       following lines, and download gettok.m from the
%       Matrix Market site.    
[head0,header]   = gettok(header);  % see note above
[head1,header]   = gettok(header);
[rep,header]     = gettok(header);
[field,header]   = gettok(header);
[symm,header]    = gettok(header);
head1 = lower(head1);
rep   = lower(rep);
field = lower(field);
symm  = lower(symm);
if ( length(symm) == 0 )
   disp(['Not enough words in header line of file ',filename]) 
   disp('Recognized format: ')
   disp('%%MatrixMarket matrix representation field symmetry')
   error('Check header line.')
end
if ( ~ strcmp(head0,'%%MatrixMarket') )
   error('Not a valid MatrixMarket header.')
end
if (  ~ strcmp(head1,'matrix') )
   disp(['This seems to be a MatrixMarket ',head1,' file.']);
   disp('This function only knows how to read MatrixMarket matrix files.');
   disp('  ');
   error('  ');
end

% Read through comments, ignoring them

commentline = fgets(mmfile);
while length(commentline) > 0 & commentline(1) == '%',
  commentline = fgets(mmfile);
end

% Read size information, then branch according to
% sparse or dense format

if ( strcmp(rep,'coordinate')) %  read matrix given in sparse 
                              %  coordinate matrix format

  [sizeinfo,count] = sscanf(commentline,'%d%d%d');
  while ( count == 0 )
     commentline =  fgets(mmfile);
     if (commentline == -1 )
       error('End-of-file reached before size information was found.')
     end
     [sizeinfo,count] = sscanf(commentline,'%d%d%d');
     if ( count > 0 & count ~= 3 )
       error('Invalid size specification line.')
     end
  end
  rows = sizeinfo(1);
  cols = sizeinfo(2);
  entries = sizeinfo(3);
  
  if  ( strcmp(field,'real') )               % real valued entries:
  
    [T,count] = fscanf(mmfile,'%f',3);
    T = [T; fscanf(mmfile,'%f')];
    if ( size(T) ~= 3*entries )
       message = ...
       str2mat('Data file does not contain expected amount of data.',...
               'Check that number of data lines matches nonzero count.');
       disp(message);
       error('Invalid data.');
    end
    T = reshape(T,3,entries)';
    A = sparse(T(:,1), T(:,2), T(:,3), rows , cols);
  
  elseif   ( strcmp(field,'complex'))            % complex valued entries:
  
    T = fscanf(mmfile,'%f',4);
    T = [T; fscanf(mmfile,'%f')];
    if ( size(T) ~= 4*entries )
       message = ...
       str2mat('Data file does not contain expected amount of data.',...
               'Check that number of data lines matches nonzero count.');
       disp(message);
       error('Invalid data.');
    end
    T = reshape(T,4,entries)';
    A = sparse(T(:,1), T(:,2), T(:,3) + T(:,4)*sqrt(-1), rows , cols);
  
  elseif  ( strcmp(field,'pattern'))    % pattern matrix (no values given):
  
    T = fscanf(mmfile,'%f',2);
    T = [T; fscanf(mmfile,'%f')];
    if ( size(T) ~= 2*entries )
       message = ...
       str2mat('Data file does not contain expected amount of data.',...
               'Check that number of data lines matches nonzero count.');
       disp(message);
       error('Invalid data.');
    end
    T = reshape(T,2,entries)';
    A = sparse(T(:,1), T(:,2), ones(entries,1) , rows , cols);

  end

elseif ( strcmp(rep,'array') ) %  read matrix given in dense 
                               %  array (column major) format

  [sizeinfo,count] = sscanf(commentline,'%d%d');
  while ( count == 0 )
     commentline =  fgets(mmfile);
     if (commentline == -1 )
       error('End-of-file reached before size information was found.')
     end
     [sizeinfo,count] = sscanf(commentline,'%d%d');
     if ( count > 0 & count ~= 2 )
       error('Invalid size specification line.')
     end
  end
  rows = sizeinfo(1);
  cols = sizeinfo(2);
  entries = rows*cols;
  if  ( strcmp(field,'real') )               % real valued entries:
    A = fscanf(mmfile,'%f',1);
    A = [A; fscanf(mmfile,'%f')];
    if ( strcmp(symm,'symmetric') | strcmp(symm,'hermitian') | strcmp(symm,'skew-symmetric') ) 
      for j=1:cols-1,
        currenti = j*rows;
        A = [A(1:currenti); zeros(j,1);A(currenti+1:length(A))];
      end
    elseif ( ~ strcmp(symm,'general') )
      disp('Unrecognized symmetry')
      disp(symm)
      disp('Recognized choices:')
      disp('   symmetric')
      disp('   hermitian')
      disp('   skew-symmetric')
      disp('   general')
      error('Check symmetry specification in header.');
    end
      A = reshape(A,rows,cols);
  elseif  ( strcmp(field,'complex'))         % complx valued entries:
    %tmpr = fscanf(mmfile,'%f',1);
    %tmpi = fscanf(mmfile,'%f',1);
    % no braindead code please..
    [tmpr,c] = fscanf(mmfile,'%f',1);
    [tmpi,c] = fscanf(mmfile,'%f',1);
    if(ne(c,1))
      fscanf(mmfile,'(');
      [tmpr,c] = fscanf(mmfile,'%f' ,1);
      fscanf(mmfile,',');
      [tmpi,c] = fscanf(mmfile,'%f',1);
      fscanf(mmfile,')');
    end
    A=reshape(zeros(entries,1) +0*i,entries,1);
    % A  = tmpr+tmpi*i;
    A(1) = tmpr+tmpi*i;
    % for j=1:entries-1
    for j=2:entries % now the algorithm is slightly different!
      [tmpr,c] = fscanf(mmfile,'%f',1);
      [tmpi,c] = fscanf(mmfile,'%f',1);
      if(ne(c,1))
        fscanf(mmfile,'(');
        [tmpr,c] = fscanf(mmfile,'%f' ,1);
        fscanf(mmfile,',');
        [tmpi,c] = fscanf(mmfile,'%f',1);
        fscanf(mmfile,')');
        end
%      A  = [A; tmpr + tmpi*i] % <- braindead code!
       A(j)=tmpr+tmpi*i;
%       printf("%f,%f",A(j));
    end
    if ( strcmp(symm,'symmetric') | strcmp(symm,'hermitian') | strcmp(symm,'skew-symmetric') ) 
      for j=1:cols-1,
        currenti = j*rows;
        A = [A(1:currenti); zeros(j,1);A(currenti+1:length(A))];
      end
    elseif ( strcmp(symm,'general') )
    %elseif ( ~ strcmp(symm,'general') )
      disp('Unrecognized symmetry')
      disp(symm)
      disp('Recognized choices:')
      disp('   symmetric')
      disp('   hermitian')
      disp('   skew-symmetric')
      disp('   general')
      error('Check symmetry specification in header.');
    end
    A = reshape(A,rows,cols);
  elseif  ( strcmp(field,'pattern'))    % pattern (makes no sense for dense)
   disp('Matrix type:',field)
   error('Pattern matrix type invalid for array storage format.');
  else                                 % Unknown matrix type
   disp('Matrix type:',field)
   error('Invalid matrix type specification. Check header against MM documentation.');
  end
end

%
% If symmetric, skew-symmetric or Hermitian, duplicate lower
% triangular part and modify entries as appropriate:
%

if ( strcmp(symm,'symmetric') )
   A = A + A.' - diag(diag(A));
   entries = nnz(A);
elseif ( strcmp(symm,'hermitian') )
   A = A + A' - diag(diag(A));
   entries = nnz(A);
elseif ( strcmp(symm,'skew-symmetric') )
   A = A - A';
   entries = nnz(A);
end

fclose(mmfile);
% Done.

