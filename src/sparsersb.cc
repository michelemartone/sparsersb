/*
 Copyright (C) 2011-2019   Michele Martone   <michelemartone _AT_ users.sourceforge.net>

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, see <http://www.gnu.org/licenses/>.
*/

/*
 * TODO wishlist (patches are welcome!):
 * adapt to when octave_idx_type is 64 bit long
 * rsb_file_vec_save (1.1)
 * all *.m files shall go to inst/
 * switch to using bootstrap.sh (instead autogen.sh) and configure.ac with environment variables, so it can be called from pkg install sparsersb-1.0.4.tar.gz
 * produce ../doc/sparsersb.txi; can use get_help_text
 * put to ./devel/ what is not to be distributed
 * make or configure should fail on missing library (actually it does not)
 * spfind in order to update easily and quickly nonzeroes
 * need A(IA,JA,VA)=nVA
 * shall add "load"; implicit filename based is confusing
 * shall rename "load"/"save" to "loadMatrixMarket"/... or something explicit
 * save/load capability (in own, rsb format)
 * should not rely on string_value().c_str()  --- stack corruption danger!
 * ("get","RSB_IO_WANT_...") is not yet available
 * (.) is incomplete. it is needed by trace()
 * (:,:) , (:,p) ... do not work, test with octave's bicg, bicgstab, cgs, ...
 * hints about how to influence caching blocking policy
 * compound_binary_op
 * for thorough testing, see Octave's test/build_sparse_tests.sh
 * introspection functionality (bytes/nnz, or  sparsersb(rsbmat,"inquire: subm") )
 * sparsersb(rsbmat,"benchmark")
 * sparsersb(rsbmat,"test")
 * minimize data copies
 * subsref, dotref, subsasgn are incomplete: need error messages there
 * in full_value(), bool arg is ignored
 * symmetry support is incomplete (scarcely defined)
 * document the semantics of the update and access operators
 * define more operators (e.g.: scaling) for 'complex'
 * create a single standard error macro for constructors
 * test sistematically all constructors
 * often missing array lenghts/type checks
 * may define as map (see is_map) so that "a.type = ..." can work
 * is_struct, find_nonzero_elem_idx  are undefined
 * are octave_triangular_conv, default_numeric_conversion_function ok ?
 * error reporting is insufficient
 * update to symmetric be forbidden or rather trigger a conversion ?
 * after file read, return various structural info
 * norm computation
 * reformat code for readability
 * warnings about incomplete complex implementation may be overzealous.
 * need matrix exponentiation through conversion to octave format.
 * Note: although librsb has been optimized for performance, sparsersb is not.
 * Note: there are dangerous casts to rsb_coo_idx_t in subsasgn: for 64-bit octave_idx_type.

 * Developer notes:
 http://www.gnu.org/software/octave/doc/interpreter/index.html
 http://www.gnu.org/software/octave/doc/interpreter/Oct_002dFiles.html#Oct_002dFiles
 http://octave.sourceforge.net/developers.html
 */

#define RSBOI_WANT_PRINT_PCT_OCTAVE_STYLE 1

#include <octave/oct.h>
#define RSBOI_USE_PATCH_OCT44 (OCTAVE_MAJOR_VERSION>=5) || ( (OCTAVE_MAJOR_VERSION==4) && (OCTAVE_MINOR_VERSION>=4))
#if RSBOI_USE_PATCH_OCT44
#include <octave/variables.h>
#include <octave/interpreter.h>
#include <octave/mach-info.h>
#endif /* RSBOI_USE_PATCH_OCT44 */
#include <octave/ov-re-mat.h>
#include <octave/ov-re-sparse.h>
#include <octave/ov-bool-sparse.h> /* RSBOI_WANT_SPMTX_SUBSREF || RSBOI_WANT_SPMTX_SUBSASGN */
#include <octave/ov-scalar.h>
#include <octave/ov-complex.h>
#include <octave/ops.h>
#include <octave/ov-typeinfo.h>
#if RSBOI_WANT_PRINT_PCT_OCTAVE_STYLE
#include <iomanip>	// std::setprecision
#endif
#include <rsb.h>

#if RSBOI_USE_PATCH_OCT44
/* transitional macros, new style */
#define RSBOI_TRY_BLK try
#define RSBOI_CATCH_BLK catch (octave::execution_exception& e) { goto err; }
#define RSBOI_IF_ERR(STMT)
#define RSBOI_IF_NERR(STMT) STMT
#define RSBOI_IF_NERR_STATE()
/* transitional macros, old style */
#else /* RSBOI_USE_PATCH_OCT44 */
#define RSBOI_IF_ERR(STMT)  if (  error_state) STMT
#define RSBOI_IF_NERR(STMT) if (! error_state) STMT
#define RSBOI_IF_NERR_STATE() if (! error_state)
#endif /* RSBOI_USE_PATCH_OCT44 */

//#define RSBOI_VERBOSE_CONFIG 1 /* poor man's trace facility */
#ifdef RSBOI_VERBOSE_CONFIG /* poor man's trace facility */
#if (RSBOI_VERBOSE_CONFIG>0)
#define RSBOI_VERBOSE RSBOI_VERBOSE_CONFIG
#endif
#endif

#define RSBOI_USE_PATCH_38143 ( defined(OCTAVE_MAJOR_VERSION) && (OCTAVE_MAJOR_VERSION>=4) ) /* See http://savannah.gnu.org/bugs/?48335#comment5 */

#if 0
#define RSBOI_WARN( MSG ) \
	octave_stdout << "Warning in "<<__func__<<"(), in file "<<__FILE__<<" at line "<<__LINE__<<":\n" << MSG;
#define RSBOI_FIXME( MSG ) RSBOI_WARN( MSG )/* new */
#else
#define RSBOI_WARN( MSG )
#endif
#define RSBOI_TODO( MSG ) RSBOI_WARN( MSG )/* new */
#define RSBOI_FIXME( MSG ) RSBOI_WARN( "FIXME: "MSG )/* new */

#define RSBOI_PRINTF( ... ) printf( __VA_ARGS__ )
#if RSBOI_VERBOSE
//printf("In file %20s (in %s) at line %10d:\n",__FILE__,__func__,__LINE__),
#define RSBOI_DEBUG_NOTICE( ... ) \
	printf("In %s(), in file %s at line %10d:\n",__func__,__FILE__,__LINE__), \
	printf( __VA_ARGS__ )
#if 0
#define RSBOI_ERROR( ... ) \
	printf("In %s(), in file %s at line %10d:\n",__func__,__FILE__,__LINE__), \
	printf( __VA_ARGS__ )
#else
#define RSBOI_ERROR( MSG ) \
	octave_stdout << "In "<<__func__<<"(), in file "<<__FILE__<<" at line "<<__LINE__<<":\n"<<MSG
#endif
#define RSBOI_DUMP RSBOI_PRINTF
#else
#define RSBOI_DUMP( ... )
#define RSBOI_DEBUG_NOTICE( ... )
#define RSBOI_ERROR( ... )
#endif
#define RSBOI_EERROR( MSG ) \
	octave_stdout << "In "<<__func__<<"(), in file "<<__FILE__<<" at line "<<__LINE__<<":\n"
#define RSBOI_TYPECODE RSB_NUMERICAL_TYPE_DOUBLE
#define RSBOI_RB RSB_DEFAULT_ROW_BLOCKING
#define RSBOI_CB RSB_DEFAULT_COL_BLOCKING
//#define RSBOI_RF RSB_FLAG_DEFAULT_STORAGE_FLAGS
#define RSBOI_RF RSB_FLAG_DEFAULT_RSB_MATRIX_FLAGS
#define RSBOI_DCF RSB_FLAG_DUPLICATES_SUM
#define RSBOI_NF RSB_FLAG_NOFLAGS
//#define RSBOI_EXPF RSB_FLAG_NOFLAGS
#define RSBOI_EXPF RSB_FLAG_IDENTICAL_FLAGS
#define RSBOI_T double
#undef RSB_FULLY_IMPLEMENTED
#define RSBOI_DESTROY(OM) {rsb_mtx_free(OM);(OM)=RSBOI_NULL;}
#define RSBOI_SOME_ERROR(ERRVAL) (ERRVAL)!=RSB_ERR_NO_ERROR
#define RSBOI_0_ERROR error
#define RSBOI_0_BADINVOERRMSG "invoking this function in the wrong way!\n"
#define RSBOI_0_ALLERRMSG "error allocating matrix!\n"
#define RSBOI_0_NOCOERRMSG "compiled without complex type support!\n"
#define RSBOI_0_NOTERRMSG "matrix is not triangular!\n"
#define RSBOI_0_ICSERRMSG "compiled with incomplete complex type support!\n"
#define RSBOI_0_EMERRMSG  "data structure is corrupt (unexpected NULL matrix pointer)!\n"
#define RSBOI_0_UNFFEMSG  "unfinished feature\n"
#define RSBOI_0_INCFERRMSG "incomplete function!\n"
#define RSBOI_0_INMISMMSG  "Index sizes of Octave differs from that of RSB:" " a conversion is needed, but yet unsupported in this version."
#define RSBOI_0_UNCFEMSG  "complex support is yet incomplete\n"
#define RSBOI_0_NEEDERR "an error condition needs to be handled, here!\n"
#define RSBOI_0_UNCBERR "matrix NOT correctly built!\n"
#define RSBOI_0_ALERRMSG  "error allocating an rsb matrix!\n"
#define RSBOI_0_FATALNBMSG  "fatal error! matrix NOT built!\n"
#define RSBOI_0_ASSERRMSG  "assignment is still unsupported on 'sparse_rsb' matrices"
#define RSBOI_0_NSQERRMSG  "matrix is not square"
#define RSBOI_0_NIYERRMSG  "not implemented yet in sparsersb"
#define RSBOI_0_INTERRMSG  "internal sparsersb error: this might be a bug -- please contact and tell us about this!"
#define RSBOI_0_INTERRMSGSTMT(STMT)  {error(RSBOI_0_INTERRMSG);STMT;}
//#define RSBOI_0_INTERRMSGSTMT(STMT)                           STMT;
#define RSBOI_D_EMPTY_MSG  ""
#define RSBOI_O_MISSIMPERRMSG  "implementation missing here\n"
#define RSBOI_O_NPMSERR  "providing non positive matrix size is not allowed!"
#define RSBOI_0_EMCHECK(M) if(!(M))RSBOI_0_ERROR(RSBOI_0_EMERRMSG);
#define RSBOI_FNSS(S)	#S
#ifndef RSB_SPARSERSB_LABEL
#define RSB_SPARSERSB_LABEL sparsersb
#endif /* RSB_SPARSERSB_LABEL */
//#define RSBOI_FNS	RSBOI_FNSS(RSB_SPARSERSB_LABEL)
#define RSBOI_FSTR	"Recursive Sparse Blocks"
#define RSBOI_FNS	"sparsersb"
#define RSBOI_LIS	"?"

#define RSBIO_DEFAULT_CORE_MATRIX  Matrix (0,0)
/* FIXME : octave_idx_type vs rsb_coo_idx_t */
#define RSBIO_NULL_STATEMENT_FOR_COMPILER_HAPPINESS {(void)1;}
#define RSBOI_OV_STRIDE 1
#define RSBOI_ZERO 0.0
//#define RSB_OI_DMTXORDER RSB_FLAG_WANT_ROW_MAJOR_ORDER
#define RSB_OI_DMTXORDER RSB_FLAG_WANT_COLUMN_MAJOR_ORDER  /* for dense matrices (multivectors) */
#define RSB_OI_TYPEINFO_STRING "rsb sparse matrix"
#define RSB_OI_TYPEINFO_TYPE    "double"

#ifdef RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX
#define RSBOI_WANT_DOUBLE_COMPLEX 1
#define ORSB_RSB_TYPE_FLAG(OBJ) (((OBJ).iscomplex())?RSB_NUMERICAL_TYPE_DOUBLE:RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX)
#else
#define RSBOI_WANT_DOUBLE_COMPLEX 0
#define ORSB_RSB_TYPE_FLAG(OBJ) RSB_NUMERICAL_TYPE_DOUBLE
#endif

#define RSBOI_USE_CXX11 ( defined(__cplusplus) && (__cplusplus>=201103L) )
#if defined(RSBOI_USE_CXX11)
#define RSBOI_NULL nullptr
#else /* RSBOI_USE_CXX11 */
#define RSBOI_NULL NULL
#endif /* RSBOI_USE_CXX11 */
#define RSBOI_INFOBUF	256
#define RSBOI_WANT_SYMMETRY 1
#define RSBOI_WANT_PRINT_DETAIL 0
#define RSBOI_WANT_PRINT_COMPLEX_OR_REAL 0
#define RSBOI_WANT_SUBSREF 1
#define RSBOI_WANT_HEAVY_DEBUG 0
#define RSBOI_WANT_VECLOAD_INSTEAD_MTX 1
#define RSBOI_WANT_MTX_LOAD 1
#define RSBOI_WANT_MTX_SAVE 1
#define RSBOI_WANT_POW 1
#define RSBOI_WANT_QSI 1 /* query string interface */
#define RSBOI_WANT_SPMTX_SUBSREF 0 /* not yet there: need to accumulate in sparse */
#define RSBOI_WANT_SPMTX_SUBSASGN 1
//#define RSBOI_PERROR(E) rsb_perror(E)
#define RSBOI_PERROR(E) if(RSBOI_SOME_ERROR(E)) rsboi_strerr(E)
#ifdef RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX
#include <octave/ov-cx-mat.h>
#include <octave/ov-cx-sparse.h>
#endif

#ifndef RSBOI_RSB_MATRIX_SOLVE
#define RSBOI_RSB_MATRIX_SOLVE(V1,V2) RSBOI_0_ERROR(RSBOI_0_NOTERRMSG)  /* any solution routine shall attached here */
#endif

#if 1
extern "C" { rsb_err_t rsb_dump_postscript_from_mtx_t(const struct rsb_mtx_t *mtxAp, rsb_blk_idx_t br, rsb_blk_idx_t bc, int width, int height, rsb_bool_t all_nnz); }
extern "C" {
rsb_err_t rsb_dump_postscript_recursion_from_mtx_t(const struct rsb_mtx_t *mtxAp, rsb_blk_idx_t br, rsb_blk_idx_t bc, int width, int height, rsb_flags_t flags, rsb_bool_t want_blocks, rsb_bool_t z_dump , rsb_bool_t want_nonzeros ); }
#endif

#if RSBOI_WANT_HEAVY_DEBUG
extern "C" {
	rsb_bool_t rsb_is_correctly_built_rcsr_matrix(const struct rsb_mtx_t *mtxAp); // forward declaration
}
#endif
#if defined(RSB_LIBRSB_VER) && (RSB_LIBRSB_VER>=10100)
extern "C" {
#if (RSB_LIBRSB_VER<=10200)
	int rsb_do_get_nnz_element(struct rsb_mtx_t *,void*,void*,void*,int);
#elif (RSB_LIBRSB_VER>=10300)
	int rsb__do_get_nnz_element(struct rsb_mtx_t *,void*,void*,void*,int);
 	#define rsb_do_get_nnz_element rsb__do_get_nnz_element
#endif
}
#endif
#if RSBOI_WANT_DOUBLE_COMPLEX
#define RSBOI_BINOP_PREVAILING_TYPE(V1,V2) (((V1).iscomplex()||(V2).iscomplex())?RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX:RSB_NUMERICAL_TYPE_DOUBLE)
#else
#define RSBOI_BINOP_PREVAILING_TYPE(V1,V2) RSBOI_TYPECODE
#endif
#if defined(RSB_LIBRSB_VER) && (RSB_LIBRSB_VER>=10100)
#define RSBOI_10100_DOCH \
"@deftypefnx {Loadable Function}      " RSBOI_FNS " (@var{S},\"render\", @var{filename}[, @var{rWidth}, @var{rHeight}])\n"\
"@deftypefnx {Loadable Function} {[@var{O} =]} " RSBOI_FNS " (@var{S},\"autotune\"[, @var{transA}, @var{nrhs}, @var{maxr}, @var{tmax}, @var{tn}, @var{sf}])\n"\

/* #define RSBOI_10100_DOC "If @var{S} is a " RSBOI_FNS " matrix and one of the \"render\",\"renderb\",\"renders\" keywords ... */
#define RSBOI_10100_DOC \
\
"If @var{S} is a " RSBOI_FNS " matrix and the \"render\" keyword is specified, and @var{filename} is a string, @var{A} will be rendered as an Encapsulated Postscript file @var{filename}. Optionally, width and height can be specified in @code{@var{rWidth}, @var{rHeight}}. Defaults are 512.\n"\
"\n"\
\
"If @var{S} is a " RSBOI_FNS " matrix and the \"autotune\" keyword is specified, autotuning of the matrix will take place, with SpMV and autotuning parameters. After the \"autotune\" string, the remaining parameters are optional. Parameter @var{transA} specifies whether to tune for untransposed (\"N\") or transposed (\"T\"); @var{nrhs} the number of right hand sides; @var{maxr} the number of tuning rounds; @var{tmax} the threads to use. If giving an output argument @var{O}, that will be assigned to the autotuned matrix, and the input one @var{A} will remain unchanged. See librsb documentation for @code{rsb_tune_spmm} to learn more.\n"
#else
#define RSBOI_10100_DOC	""
#define RSBOI_10100_DOCH	""
#endif

#define RSBOI_VERSION	100007	/* e.g. 100007 means 1.0.7 */

#if defined(USE_64_BIT_IDX_T) || defined(OCTAVE_ENABLE_64) || defined(RSBOI_DETECTED_LONG_IDX) /* 4.1.0+ / 4.0.3 / any */
#define RSBOI_O64_R32 1
#else /* USE_64_BIT_IDX_T */
#define RSBOI_O64_R32 0
#endif /* USE_64_BIT_IDX_T */

#define RSBOI_SIZEMAX RSB_MAX_MATRIX_DIM /* Upper limit to librsb matrix dimension. */

static rsb_err_t rsboi_idxv_overflow( const idx_vector & IM, const idx_vector & JM)
{
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	
	if( IM.extent(0) > RSBOI_SIZEMAX || JM.extent(0) > RSBOI_SIZEMAX )
		errval = RSB_ERR_LIMITS;

	return errval;
}

#if RSBOI_O64_R32
static rsb_err_t rsboi_idx_overflow( rsb_err_t *errvalp, octave_idx_type idx1, octave_idx_type idx2=0, octave_idx_type idx3=0)
{
	rsb_err_t errval = RSB_ERR_NO_ERROR;

	if( idx1 > RSBOI_SIZEMAX || idx2 > RSBOI_SIZEMAX || idx3 > RSBOI_SIZEMAX )
		errval = RSB_ERR_LIMITS;
	if( errvalp )
		*errvalp = errval;

	return errval;
}

static void rsboi_oi2ri( octave_idx_type * IP, rsb_nnz_idx_t nnz)
{
	// octave_idx_type -> rsb_coo_idx_t
	rsb_coo_idx_t * RP = (rsb_coo_idx_t *) IP;

	const octave_idx_type * OP = (const octave_idx_type*) IP;
	rsb_nnz_idx_t nzi;

	for(nzi=0;nzi<nnz;++nzi)
		RP[nzi] = OP[nzi];
}

static void rsboi_ri2oi( rsb_coo_idx_t * IP, rsb_nnz_idx_t nnz)
{
	// rsb_coo_idx_t -> octave_idx_type
	const rsb_coo_idx_t * RP = (const rsb_coo_idx_t *) IP;

	octave_idx_type * OP = (octave_idx_type*) IP;
	rsb_nnz_idx_t nzi;

	for(nzi=0;nzi<nnz;++nzi)
		OP[nnz-(nzi+1)]=RP[nnz-(nzi+1)];
}

static rsb_err_t rsboi_mtx_get_coo(const struct rsb_mtx_t *mtxAp, void * VA, octave_idx_type * IA, octave_idx_type * JA, rsb_flags_t flags )
{
	// assumes tacitly that rsboi_idx_overflow(IA[i],JA[i])==false for i in 0..nnzA-1.
	rsb_err_t errval = RSB_ERR_NO_ERROR;

	errval = rsb_mtx_get_coo(mtxAp, VA, (rsb_coo_idx_t *)IA, (rsb_coo_idx_t*)JA, flags );
	rsb_nnz_idx_t nnzA = 0;
	rsb_mtx_get_info(mtxAp,RSB_MIF_MATRIX_NNZ__TO__RSB_NNZ_INDEX_T,&nnzA); // FIXME: make this a member and use nnz()
	rsboi_ri2oi((rsb_coo_idx_t *)IA,nnzA);
	rsboi_ri2oi((rsb_coo_idx_t *)JA,nnzA);

	return errval;
}

static struct rsb_mtx_t *rsboi_mtx_alloc_from_csc_const(const void *VA, /*const*/ octave_idx_type * IA, /*const*/ octave_idx_type * CP, rsb_nnz_idx_t nnzA, rsb_type_t typecode, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_blk_idx_t brA, rsb_blk_idx_t bcA, rsb_flags_t flagsA, rsb_err_t *errvalp)
{
	struct rsb_mtx_t *mtxAp = RSBOI_NULL;

	if( RSBOI_SOME_ERROR(rsboi_idx_overflow(errvalp,nrA,ncA,nnzA) ) )
		goto ret;
	rsboi_oi2ri(IA,nnzA);
	rsboi_oi2ri(CP,ncA+1);
	mtxAp = rsb_mtx_alloc_from_csc_const(VA, (rsb_coo_idx_t *)IA, (rsb_coo_idx_t *)CP, nnzA, typecode, nrA, ncA, brA, bcA, flagsA, errvalp);
	rsboi_ri2oi((rsb_coo_idx_t *)IA,nnzA);
	rsboi_ri2oi((rsb_coo_idx_t *)CP,ncA+1);
ret:
	return mtxAp;
}

static struct rsb_mtx_t *rsboi_mtx_alloc_from_coo_const(const void *VA, /*const*/ octave_idx_type * IA, /*const*/ octave_idx_type * JA, rsb_nnz_idx_t nnzA, rsb_type_t typecode, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_blk_idx_t brA, rsb_blk_idx_t bcA, rsb_flags_t flagsA, rsb_err_t *errvalp)
{
	struct rsb_mtx_t *mtxAp = RSBOI_NULL;

	if( RSBOI_SOME_ERROR(rsboi_idx_overflow(errvalp,nrA,ncA,nnzA) ) )
		goto ret;
	rsboi_oi2ri(IA,nnzA);
	rsboi_oi2ri(JA,nnzA);
	mtxAp = rsb_mtx_alloc_from_coo_const(VA, (rsb_coo_idx_t *)IA, (rsb_coo_idx_t *)JA, nnzA, typecode, nrA, ncA, brA, bcA, flagsA, errvalp);
	rsboi_ri2oi((rsb_coo_idx_t *)IA,nnzA);
	rsboi_ri2oi((rsb_coo_idx_t *)JA,nnzA);
ret:
	return mtxAp;
}
#else /* RSBOI_O64_R32 */
static rsb_err_t rsboi_mtx_get_coo(const struct rsb_mtx_t *mtxAp, void * VA, octave_idx_type * IA, octave_idx_type * JA, rsb_flags_t flags )
{
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	errval = rsb_mtx_get_coo(mtxAp, VA, IA, JA, flags );
	return errval;
}

static struct rsb_mtx_t *rsboi_mtx_alloc_from_csc_const(const void *VA, const octave_idx_type * IA, const octave_idx_type * CP, rsb_nnz_idx_t nnzA, rsb_type_t typecode, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_blk_idx_t brA, rsb_blk_idx_t bcA, rsb_flags_t flagsA, rsb_err_t *errvalp)
{
	struct rsb_mtx_t *mtxAp = RSBOI_NULL;
	mtxAp = rsb_mtx_alloc_from_csc_const(VA, IA, CP, nnzA, typecode, nrA, ncA, brA, bcA, flagsA, errvalp);
	return mtxAp;
}

static struct rsb_mtx_t *rsboi_mtx_alloc_from_coo_const(const void *VA, const octave_idx_type * IA, const octave_idx_type * JA, rsb_nnz_idx_t nnzA, rsb_type_t typecode, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_blk_idx_t brA, rsb_blk_idx_t bcA, rsb_flags_t flagsA, rsb_err_t *errvalp)
{
	struct rsb_mtx_t *mtxAp = RSBOI_NULL;
	mtxAp = rsb_mtx_alloc_from_coo_const(VA, IA, JA, nnzA, typecode, nrA, ncA, brA, bcA, flagsA, errvalp);
	return mtxAp;
}
#endif /* RSBOI_O64_R32 */

void rsboi_strerr(rsb_err_t errval)
{
	const int errstrlen = 128;
	char errstr[errstrlen];

	rsb_strerror_r(errval,errstr,errstrlen);
	octave_stdout<<"librsb error:"<<errstr<<"\n";
}

struct rsboi_coo_matrix_t
{
	octave_idx_type * IA, * JA;	 /** row and columns indices */
	octave_idx_type nrA,ncA;	 /** matrix (declared) nonzeros */
	octave_idx_type nnzA;		 /** matrix rows, columns */
	void * VA;			 /** values of data elements */
	rsb_type_t typecode;		 /** as specified in the RSB_NUMERICAL_TYPE_* preprocessor symbols in types.h 	*/
};

static const RSBOI_T rsboi_pone[] = {+1.0,0.0};
static const RSBOI_T rsboi_mone[] = {-1.0,0.0};
static const RSBOI_T rsboi_zero[] = { 0.0,0.0}; /* two elements, as shall work also with complex */

static octave_base_value *default_numeric_conversion_function (const octave_base_value& a);

static bool sparsersb_tester(void)
{
#if (RSBOI_VERSION < 100002)	
	if(sizeof(octave_idx_type)!=sizeof(rsb_coo_idx_t))
	{
		RSBOI_ERROR(RSBOI_0_INMISMMSG);
		goto err;
	}
#else /* RSBOI_VERSION */
	if(sizeof(octave_idx_type)< sizeof(rsb_coo_idx_t))
	{
		RSBOI_ERROR(RSBOI_0_INMISMMSG);
		goto err;
	}
#endif /* RSBOI_VERSION */
	RSBOI_WARN(RSBOI_0_INCFERRMSG);
	return true;
err:
	return false;
}

static bool rsboi_sparse_rsb_loaded = false;

class octave_sparsersb_mtx : public octave_sparse_matrix
{
	private:
	public:
	struct rsb_mtx_t *mtxAp;
	public:
		octave_sparsersb_mtx (void) : octave_sparse_matrix(RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			this->mtxAp = RSBOI_NULL;
		}

		octave_sparsersb_mtx (const octave_sparse_matrix &sm) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		}

#if RSBOI_WANT_MTX_LOAD
		octave_sparsersb_mtx (const std::string &mtxfilename, rsb_type_t typecode = RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			rsb_err_t errval = RSB_ERR_NO_ERROR;

			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			if(!(this->mtxAp = rsb_file_mtx_load(mtxfilename.c_str(),RSBOI_RF,typecode,&errval)))
#if RSBOI_WANT_VECLOAD_INSTEAD_MTX
				/* no problem */;
#else
				RSBOI_ERROR(RSBOI_0_ALERRMSG);
			RSBOI_PERROR(errval);
			if(!this->mtxAp)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
#endif
		}
#endif

		//void alloc_rsb_mtx_from_coo_copy(const idx_vector &IM, const idx_vector &JM, const void * SMp, octave_idx_type nrA, octave_idx_type ncA, bool iscomplex=false, rsb_flags_t eflags=RSBOI_DCF)
		void alloc_rsb_mtx_from_coo_copy(idx_vector & IM, idx_vector & JM, const void * SMp, octave_idx_type nrA, octave_idx_type ncA, bool iscomplex=false, rsb_flags_t eflags=RSBOI_DCF)
		{
			octave_idx_type nnzA = IM.length();
			rsb_err_t errval = RSB_ERR_NO_ERROR;
#if RSBOI_WANT_DOUBLE_COMPLEX
			rsb_type_t typecode = iscomplex?RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX:RSB_NUMERICAL_TYPE_DOUBLE;
#else /* RSBOI_WANT_DOUBLE_COMPLEX */
			rsb_type_t typecode = RSBOI_TYPECODE;
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
			const octave_idx_type *IA = RSBOI_NULL,*JA = RSBOI_NULL;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
#if RSBOI_WANT_SYMMETRY
			/* shall verify if any symmetry is present */
#endif

			IA = (const octave_idx_type*)IM.raw();
		       	JA = (const octave_idx_type*)JM.raw();

			//RSB_DO_FLAG_ADD(eflags,rsb_util_determine_uplo_flags(IA,JA,nnzA));
			
			if( (nrA==0 || ncA==0) && RSBOI_SOME_ERROR(errval=rsboi_idxv_overflow( IM, JM )))
				goto err;

			if(!(this->mtxAp = rsboi_mtx_alloc_from_coo_const(SMp,(octave_idx_type*)IA,(octave_idx_type*)JA,nnzA,typecode,nrA,ncA,RSBOI_RB,RSBOI_CB,RSBOI_RF|eflags ,&errval)))
				RSBOI_ERROR(RSBOI_0_ALERRMSG);
			//RSBOI_MP(this->mtxAp);
err:
			RSBOI_PERROR(errval);
			if(!this->mtxAp)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

#if RSBOI_WANT_DOUBLE_COMPLEX
		octave_sparsersb_mtx (idx_vector &IM, idx_vector &JM, const ComplexMatrix &SM,
			octave_idx_type nrA, octave_idx_type ncA, rsb_flags_t eflags) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			this->alloc_rsb_mtx_from_coo_copy(IM,JM,SM.data(),nrA,ncA,true,eflags);
		}
#endif

		octave_sparsersb_mtx (idx_vector &IM, idx_vector &JM, const Matrix &SM,
			octave_idx_type nrA, octave_idx_type ncA, rsb_flags_t eflags) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			this->alloc_rsb_mtx_from_coo_copy(IM,JM,SM.data(),nrA,ncA,false,eflags);
		}

		void alloc_rsb_mtx_from_csc_copy(const SparseMatrix &sm)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rsb_nnz_idx_t nnzA = 0;
			Array<rsb_coo_idx_t> IA( dim_vector(1,sm.nnz()) );
			Array<rsb_coo_idx_t> JA( dim_vector(1,sm.nnz()) );
			rsb_err_t errval = RSB_ERR_NO_ERROR;
			/* bool islowtri=sm.is_lower_triangular(),isupptri=sm.is_upper_triangular(); */
			rsb_flags_t eflags = RSBOI_RF;
			rsb_type_t typecode = RSB_NUMERICAL_TYPE_DOUBLE;
			octave_idx_type nrA = sm.rows (), ncA = sm.cols ();

#if RSBOI_WANT_SYMMETRY
			if(sm.issymmetric())
				RSB_DO_FLAG_ADD(eflags,RSB_FLAG_LOWER_SYMMETRIC|RSB_FLAG_TRIANGULAR);
#endif
			if(!(this->mtxAp = rsboi_mtx_alloc_from_csc_const(sm.data(),sm.ridx(),sm.cidx(), nnzA=sm.nnz(),typecode, nrA, ncA, RSBOI_RB, RSBOI_CB, eflags,&errval)))
				RSBOI_ERROR(RSBOI_0_ALLERRMSG);
			RSBOI_PERROR(errval);
			if(!this->mtxAp)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparsersb_mtx (const Matrix &m) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			SparseMatrix sm(m);
			this->alloc_rsb_mtx_from_csc_copy(sm);
		}

#if RSBOI_WANT_DOUBLE_COMPLEX
		void alloc_rsb_mtx_from_csc_copy(const SparseComplexMatrix &sm)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			octave_idx_type nrA = sm.rows ();
			octave_idx_type ncA = sm.cols ();
			octave_idx_type nnzA = 0;
			Array<rsb_coo_idx_t> IA( dim_vector(1,sm.nnz()) );
			Array<rsb_coo_idx_t> JA( dim_vector(1,sm.nnz()) );
			rsb_err_t errval = RSB_ERR_NO_ERROR;
			/* bool islowtri=sm.is_lower_triangular(),isupptri=sm.is_upper_triangular(); */
			rsb_flags_t eflags = RSBOI_RF;
			rsb_type_t typecode = RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX;

#if RSBOI_WANT_SYMMETRY
			if(sm.ishermitian())
				RSB_DO_FLAG_ADD(eflags,RSB_FLAG_LOWER_HERMITIAN|RSB_FLAG_TRIANGULAR);
#endif
			if(!(this->mtxAp = rsboi_mtx_alloc_from_csc_const(sm.data(),sm.ridx(),sm.cidx(), nnzA=sm.nnz(),typecode, nrA, ncA, RSBOI_RB, RSBOI_CB, eflags,&errval)))
				RSBOI_ERROR(RSBOI_0_ALLERRMSG);
			RSBOI_PERROR(errval);
			if(!this->mtxAp)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparsersb_mtx (const ComplexMatrix &cm) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			this->alloc_rsb_mtx_from_csc_copy(SparseComplexMatrix(cm));
		}

		octave_sparsersb_mtx (const SparseComplexMatrix &sm, rsb_type_t typecode = RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			this->alloc_rsb_mtx_from_csc_copy(sm);
		}
#endif

		octave_sparsersb_mtx (const SparseMatrix &sm, rsb_type_t typecode = RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			this->alloc_rsb_mtx_from_csc_copy(sm);
		}

		octave_sparsersb_mtx (struct rsb_mtx_t *mtxBp) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX), mtxAp(mtxBp)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			if(!this->mtxAp)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparsersb_mtx (const octave_sparsersb_mtx& T) :
		octave_sparse_matrix (T)  {
			rsb_err_t errval = RSB_ERR_NO_ERROR;
			struct rsb_mtx_t *mtxBp = RSBOI_NULL;

		       	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			errval = rsb_mtx_clone(&mtxBp,RSB_NUMERICAL_TYPE_SAME_TYPE,RSB_TRANSPOSITION_N,RSBOI_NULL,T.mtxAp,RSBOI_EXPF);
			this->mtxAp = mtxBp;
		};
		octave_idx_type length (void) const { return this->nnz(); }
		octave_idx_type nelem (void) const { return this->nnz(); }
		octave_idx_type numel (void) const { return this->nnz(); }
		octave_idx_type nnz (void) const { rsb_nnz_idx_t nnzA = 0; RSBOI_0_EMCHECK(this->mtxAp); rsb_mtx_get_info(this->mtxAp,RSB_MIF_MATRIX_NNZ__TO__RSB_NNZ_INDEX_T,&nnzA);  return nnzA;}
		dim_vector dims (void) const { return (dim_vector(this->rows(),this->cols())); }
		octave_idx_type dim1 (void) const { return this->rows(); }
		octave_idx_type dim2 (void) const { return this->cols(); }
		octave_idx_type rows (void) const { rsb_coo_idx_t Anr=0; RSBOI_0_EMCHECK(this->mtxAp); rsb_mtx_get_info(this->mtxAp,RSB_MIF_MATRIX_ROWS__TO__RSB_COO_INDEX_T,&Anr);  return Anr;}
		octave_idx_type cols (void) const { rsb_coo_idx_t Anc=0; RSBOI_0_EMCHECK(this->mtxAp); rsb_mtx_get_info(this->mtxAp,RSB_MIF_MATRIX_COLS__TO__RSB_COO_INDEX_T,&Anc);  return Anc;}
		rsb_flags_t rsbflags(void) const { rsb_flags_t Aflags=0; RSBOI_0_EMCHECK(this->mtxAp); rsb_mtx_get_info(this->mtxAp,RSB_MIF_MATRIX_FLAGS__TO__RSB_FLAGS_T,&Aflags);  return Aflags;}
		rsb_type_t rsbtype(void) const { rsb_type_t Atype=0; RSBOI_0_EMCHECK(this->mtxAp); rsb_mtx_get_info(this->mtxAp,RSB_MIF_MATRIX_TYPECODE__TO__RSB_TYPE_T,&Atype);  return Atype;}
		//octave_idx_type rows (void) const { RSBOI_0_EMCHECK(this->mtxAp);return this->mtxAp->nrA; }
		//octave_idx_type cols (void) const { RSBOI_0_EMCHECK(this->mtxAp);return this->mtxAp->ncA; }
		octave_idx_type columns (void) const { return this->cols(); }
		octave_idx_type nzmax (void) const { return this->nnz(); }
		octave_idx_type capacity (void) const { return this->nnz(); }
		size_t byte_size (void) const { RSBOI_0_EMCHECK(this->mtxAp);size_t so=0;rsb_mtx_get_info(this->mtxAp,RSB_MIF_TOTAL_SIZE__TO__SIZE_T,&so);return so; }

		virtual ~octave_sparsersb_mtx (void)
		{
			RSBOI_DEBUG_NOTICE("destroying librsb matrix %p\n",this->mtxAp);
			RSBOI_DESTROY(this->mtxAp);
		}

		virtual octave_base_value *clone (void) const
		{
			RSBOI_DEBUG_NOTICE("cloning librsb matrix %p\n",this->mtxAp);
			return new octave_sparsersb_mtx (*this);
		}

		virtual octave_base_value *empty_clone (void) const
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			return new octave_sparsersb_mtx ();
		}

		virtual SparseMatrix sparse_matrix_value(bool = false)const
		{
			struct rsboi_coo_matrix_t rcm;
			rsb_err_t errval = RSB_ERR_NO_ERROR;
			rsb_nnz_idx_t nnzA,nzi;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			RSBOI_0_EMCHECK(this->mtxAp);
			nnzA = this->nnz();
			Array<octave_idx_type> IA( dim_vector(1,nnzA) );
			Array<octave_idx_type> JA( dim_vector(1,nnzA) );
			Array<RSBOI_T> VA( dim_vector(1,nnzA) );

			rcm.IA = (octave_idx_type*)IA.data(),rcm.JA = (octave_idx_type*)JA.data();
			if(!this->is_real_type())
			{
				Array<Complex> VAC( dim_vector(1,nnzA) );
				RSBOI_T* VAp = ((RSBOI_T*)VA.data());
				rcm.VA = (RSBOI_T*)VAC.data();
#if RSBOI_WANT_SYMMETRY
				/* FIXME: and now ? shall we expand symmetry or not ? */
#endif
				/* FIXME: shall use some librsb's dedicated call for this */
				errval = rsboi_mtx_get_coo(this->mtxAp,rcm.VA,rcm.IA,rcm.JA,RSB_FLAG_C_INDICES_INTERFACE);
				for(nzi=0;nzi<nnzA;++nzi)
					VAp[nzi]=((RSBOI_T*)rcm.VA)[2*nzi];
			}
			else
			{
				rcm.VA = (RSBOI_T*)VA.data();
				errval = rsboi_mtx_get_coo(this->mtxAp,rcm.VA,rcm.IA,rcm.JA,RSB_FLAG_C_INDICES_INTERFACE);
			}
			rcm.nrA = this->rows();
			rcm.ncA = this->cols();

			return SparseMatrix(VA,IA,JA,rcm.nrA,rcm.ncA);
		}

		virtual Matrix matrix_value(bool = false)const
		{
			RSBOI_FIXME("inefficient!");
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			Matrix cm = this->sparse_matrix_value().matrix_value();
			return cm;
		}

		virtual Matrix full_sym_real_value()const
		{
			// Conversion to full, with symmetry expansion.
			RSBOI_FIXME("inefficient (see transpose)!");
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

			const octave_idx_type rn = this->rows(), cn = this->cols();
			Matrix v2(rn,cn,RSBOI_ZERO);
			octave_value retval = v2;
			rsb_err_t errval = RSB_ERR_NO_ERROR;
			errval |= rsb_mtx_add_to_dense(&rsboi_pone,this->mtxAp,rn,rn,cn,RSB_BOOL_TRUE,(RSBOI_T*)v2.data());
			for(int i = 0; i<rn; ++i)
				v2(i,i) = RSBOI_ZERO;
			v2 = v2.transpose();
			errval |= rsb_mtx_add_to_dense(&rsboi_pone,this->mtxAp,rn,rn,cn,RSB_BOOL_TRUE,(RSBOI_T*)v2.data());
			if(RSBOI_SOME_ERROR(errval))
				RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);
			return v2;
		} /* full_sym_real_value */

		virtual ComplexMatrix full_sym_cplx_value()const
		{
			// Conversion to full, with symmetry expansion.
			RSBOI_FIXME("inefficient (see transpose)!");
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

			const octave_idx_type rn = this->rows(), cn = this->cols();
			ComplexMatrix v2(rn,cn,RSBOI_ZERO);
			octave_value retval = v2;
			rsb_err_t errval = RSB_ERR_NO_ERROR;
			errval |= rsb_mtx_add_to_dense(&rsboi_pone,this->mtxAp,rn,rn,cn,RSB_BOOL_TRUE,(RSBOI_T*)v2.data());
			for(int i = 0; i<rn; ++i)
				v2(i,i) = RSBOI_ZERO;
			v2 = v2.transpose();
			errval |= rsb_mtx_add_to_dense(&rsboi_pone,this->mtxAp,rn,rn,cn,RSB_BOOL_TRUE,(RSBOI_T*)v2.data());
			if(RSBOI_SOME_ERROR(errval))
				RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);
			return v2;
		} /* full_sym_cplx_value */

		virtual octave_value full_value(void)const
		{
			RSBOI_FIXME("inefficient!");
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

			if(is__symmetric() || is__hermitian())
			{
				if(this->is_real_type())
					return this->full_sym_real_value();
				else
					return this->full_sym_cplx_value();
			}
			else
			{
				if(this->is_real_type())
					return this->matrix_value();
				else
					return this->complex_matrix_value();
			}
		}

#if RSBOI_WANT_DOUBLE_COMPLEX
		virtual ComplexMatrix complex_matrix_value(bool = false)const
		{
			RSBOI_FIXME("inefficient!");
			octave_sparse_complex_matrix ocm = this->sparse_complex_matrix_value();
			ComplexMatrix cm = ocm.complex_matrix_value();
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			return cm;
		}

		virtual SparseComplexMatrix sparse_complex_matrix_value(bool = false)const
		{
			struct rsboi_coo_matrix_t rcm;
			rsb_err_t errval = RSB_ERR_NO_ERROR;
			rsb_nnz_idx_t nnzA,nzi;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			RSBOI_0_EMCHECK(this->mtxAp);
			nnzA = this->nnz();
			Array<octave_idx_type> IA( dim_vector(1,nnzA) );
			Array<octave_idx_type> JA( dim_vector(1,nnzA) );
			Array<Complex> VA( dim_vector(1,nnzA) );
			RSBOI_T* VAp = ((RSBOI_T*)VA.data());

			rcm.IA = (octave_idx_type*)IA.data(),rcm.JA = (octave_idx_type*)JA.data();
			rcm.VA = VAp;
			errval = rsboi_mtx_get_coo(this->mtxAp,rcm.VA,rcm.IA,rcm.JA,RSB_FLAG_C_INDICES_INTERFACE);
#if RSBOI_WANT_SYMMETRY
			/* FIXME: and now ? shall we expand symmetry or not ? */
#endif
			/* FIXME: shall use some librsb's dedicated call for this */
			if(this->is_real_type())
				for(nzi=0;nzi<nnzA;++nzi)
					VAp[2*(nnzA-1-nzi)+0]=VAp[(nnzA-1-nzi)+0],
					VAp[2*(nnzA-1-nzi)+1]=0;
			rcm.nrA = this->rows();
			rcm.ncA = this->cols();

			return SparseComplexMatrix(VA,IA,JA,rcm.nrA,rcm.ncA);
		}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

		//octave_value::assign_op, int, int, octave_value (&)(const octave_base_value&, const octave_base_value&)
		//octave_value::assign_op, int, int, octave_value (&)
		//octave_value  assign_op (const octave_base_value&, const octave_base_value&) {}
		// octave_value::assign_op octave_value::binary_op_to_assign_op (binary_op op) { assign_op retval; return retval; }

#if RSBOI_WANT_SUBSREF
		octave_value do_index_op(const octave_value_list& idx, bool resize_ok = false)
		{
				RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
				rsb_err_t errval = RSB_ERR_NO_ERROR;
				octave_value retval;
				// octave_idx_type n_idx = idx.length ();

				//if (type.length () == 1)
				{

  					octave_idx_type n_idx = idx.length ();
					if (n_idx == 1 )
					{
						RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
#if RSBOI_WANT_SPMTX_SUBSREF
						octave_value_list ovl = idx;
						if(ovl(0).issparse())
						{
  							SparseBoolMatrix sm = SparseBoolMatrix (ovl(0).sparse_matrix_value());
							octave_idx_type * ir = sm.mex_get_ir ();
							octave_idx_type * jc = sm.mex_get_jc ();
					        	octave_idx_type nr = sm.rows ();
        						octave_idx_type nc = sm.cols ();
							RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

        						for (octave_idx_type j = 0; j < nc; j++)
							{
							  std::cout << jc[j] << ".." << jc[j+1] << "\n";
        						  for (octave_idx_type i = jc[j]; i < jc[j+1]; i++)
							  {
							    std::cout << ir[i] << " " << j << "\n";
							  }
							}
							RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
							retval = octave_value(this->clone()); // matches but .. heavy ?!
						}
						else
#endif /* RSBOI_WANT_SPMTX_SUBSREF */
						{
	    					idx_vector i = idx (0).index_vector ();
#if   defined(RSB_LIBRSB_VER) && (RSB_LIBRSB_VER< 10100)
						octave_idx_type ii = i(0);
						RSBOI_ERROR("");
#elif defined(RSB_LIBRSB_VER) && (RSB_LIBRSB_VER>=10100)
						octave_idx_type ii = i(0);
						RSBOI_DEBUG_NOTICE("get_element (%d)\n",ii);
						if(is_real_type())
						{
							RSBOI_T rv;
							errval = rsb_do_get_nnz_element(this->mtxAp,&rv,RSBOI_NULL,RSBOI_NULL,ii);
							retval = rv;
						}
						else
						{
							Complex rv;
							errval = rsb_do_get_nnz_element(this->mtxAp,&rv,RSBOI_NULL,RSBOI_NULL,ii);
							retval = rv;
						}
						if(RSBOI_SOME_ERROR(errval))
						{
							if(ii>=this->nnz() || ii<0)
								error ("trying accessing element %ld: index out of bounds !",(long int)ii+1);
							else
								error ("trying accessing element %ld: this seems bug!",(long int)ii+1);
						}
#endif
						}
					}
					else
					if (n_idx == 2 )
	  				{
					RSBOI_TRY_BLK
					{
	    					idx_vector i = idx (0).index_vector ();
						RSBOI_IF_NERR_STATE()
	      					{
#if RSBOI_WANT_SYMMETRY
							/* FIXME: and now ? */
#endif
							if(is_real_type())
							{
								idx_vector j = idx (1).index_vector ();
								RSBOI_T rv;
						  		rsb_coo_idx_t ii = -1, jj = -1;
  								ii = i(0); jj = j(0);
								RSBOI_DEBUG_NOTICE("get_elements (%d %d)\n",ii,jj);
       								errval = rsb_mtx_get_values(this->mtxAp,&rv,&ii,&jj,1,RSBOI_NF);
								retval = rv;
								RSBOI_IF_NERR(;)
							}
							else
							{
								idx_vector j = idx (1).index_vector ();
								Complex rv;
						  		rsb_coo_idx_t ii =-1, jj = -1;
  								ii = i(0); jj = j(0);
								RSBOI_DEBUG_NOTICE("get_elements (%d %d) complex\n",ii,jj);
       								errval = rsb_mtx_get_values(this->mtxAp,&rv,&ii,&jj,1,RSBOI_NF);
								retval = rv;
								RSBOI_IF_NERR(;)
							}
	      					}
					}
					RSBOI_CATCH_BLK
	  				}
				}
err:
				return retval;
		}

		octave_value subsref (const std::string &type, const std::list<octave_value_list>& idx)
		{
			octave_value retval;
			int skip = 1;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rsb_err_t errval = RSB_ERR_NO_ERROR;

			RSBOI_TRY_BLK
			{
			switch (type[0])
			{
				case '(':
					retval = do_index_op(idx.front());
				break;

				case '.':
					RSBOI_DEBUG_NOTICE("UNFINISHED\n");
					break;

				case '{':
					error ("%s cannot be indexed with %c", type_name().c_str(), type[0]);
					break;

				default:
					panic_impossible ();
			}
			}
			RSBOI_CATCH_BLK
			RSBOI_IF_NERR(
				retval = retval.next_subsref (type, idx, skip);
				)
err:
			return retval;
		} /* subsref */
#else /* RSBOI_WANT_SUBSREF */
		/* FIXME: need an alternative, bogus implementation of subsref */
#endif /* RSBOI_WANT_SUBSREF */

		octave_value_list dotref (const octave_value_list& idx)
		{
			octave_value_list retval;

			std::string nm = idx(0).string_value ();
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

			/*    if (nm == "type")
				  if (isupper ())
				retval = octave_value ("Upper");
				  else
				retval = octave_value ("Lower");
				else*/
			error ("%s can indexed with .%s",
				type_name().c_str(), nm.c_str());

			return retval;
		}

		bool is_map (void) const { return true; }
		bool issparse(void) const { RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return true; }
		bool is_real_type (void) const { RSBOI_0_EMCHECK(this->mtxAp); RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return this->rsbtype()==RSB_NUMERICAL_TYPE_DOUBLE?true:false; }
		bool is_diagonal (void) const { RSBOI_0_EMCHECK(this->mtxAp); RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_DIAGONAL)?true:false; }/* FIXME: new: not sure whether this is ever called */
		bool is_lower_triangular (void) const { RSBOI_0_EMCHECK(this->mtxAp); RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_LOWER_TRIANGULAR)?true:false; }/* FIXME: new: not sure whether this is ever called */
		bool is_upper_triangular (void) const { RSBOI_0_EMCHECK(this->mtxAp); RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_UPPER_TRIANGULAR)?true:false; }/* FIXME: new: not sure whether this is ever called */
		bool iscomplex (void) const { RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG); return !is_real_type(); }
		bool isreal (void) const { RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG); return  is_real_type(); }
		bool is_bool_type (void) const { return false; }
		bool isinteger (void) const { return false; }
		bool is_square (void) const { return this->rows()==this->cols(); }
		bool is_empty (void) const { return false; }
		bool is__symmetric (void) const { if(RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_SYMMETRIC))return true; return false; }
		bool is__hermitian (void) const { if(RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_HERMITIAN))return true; return false; }
		std::string get_symmetry (void) const { return (RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_SYMMETRIC)?"S": (RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_HERMITIAN)?"H":"U")); }
		bool is__triangular (void) const
	       	{
			rsb_bool_t retval = RSB_BOOL_FALSE;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

		       	if(!this->mtxAp)
			       	retval = RSB_BOOL_FALSE;
			else
#if RSBOI_WANT_SYMMETRY
		       	if( (!RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_SYMMETRIC)) || RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_DIAGONAL) )
#endif
				retval = RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_TRIANGULAR)?RSB_BOOL_TRUE:RSB_BOOL_FALSE;
			return retval;
		}
//		int is_struct (void) const { return false; }

		bool save_ascii (std::ostream& os)
		{
			error("save_ascii() " RSBOI_0_NIYERRMSG);
			return false;
		}
		bool load_ascii (std::istream& is)
		{
			error("load_ascii() " RSBOI_0_NIYERRMSG);
			return false;
		}
		bool save_binary (std::ostream& os, bool& save_as_floats)
		{
			error("save_binary() " RSBOI_0_NIYERRMSG);
			return false;
		}
#if RSBOI_USE_PATCH_OCT44
		bool load_binary (std::istream& is, bool swap, octave::mach_info::float_format fmt)
#else /* RSBOI_USE_PATCH_OCT44 */
		// would break on octave6
		bool load_binary (std::istream& is, bool swap, oct_mach_info::float_format fmt)
#endif /* RSBOI_USE_PATCH_OCT44 */
		{
			error("load_binary() " RSBOI_0_NIYERRMSG);
			return false;
		}
		octave_value subsasgn (const std::string& type, const std::list<octave_value_list>& idx, const octave_value& rhs)
		{
			octave_value retval;
#if 0
			rsb_err_t errval = RSB_ERR_NO_ERROR;
#endif
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

			switch (type[0])
			{

				case '(':
				{
				if (type.length () == 1)
				{
					//retval = numeric_assign (type, idx, rhs);
					//RSBOI_DEBUG_NOTICE("UNFINISHED\n");
					octave_idx_type n_idx = idx.front().length ();
					switch (n_idx)
    					{
						case 0:
						retval = matrix;
						RSBOI_DEBUG_NOTICE("UNFINISHED\n");
						break;
						case 1:
						{
#if RSBOI_WANT_SPMTX_SUBSASGN
					{
						RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
						octave_value_list ovl = idx.front();
						if(ovl(0).issparse() && ovl(0).isreal() && rhs.isreal())
						{
  							SparseBoolMatrix sm = SparseBoolMatrix (ovl(0).sparse_matrix_value());
							octave_idx_type * ir = sm.mex_get_ir ();
							octave_idx_type * jc = sm.mex_get_jc ();
					        	octave_idx_type nr = sm.rows ();
        						octave_idx_type nc = sm.cols ();
							RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
							RSBOI_T rv = rhs.double_value();

        						for (octave_idx_type j = 0; j < nc; j++)
							{
        						  for (octave_idx_type i = jc[j]; i < jc[j+1]; i++)
							  {
							    rsb_err_t errval = RSB_ERR_NO_ERROR;
							    rsb_coo_idx_t ii = static_cast<rsb_coo_idx_t>(ir[i]); // Note: potentioally dangerous casts, if types are different and matrix huge.
							    rsb_coo_idx_t jj = static_cast<rsb_coo_idx_t>(j);

							    errval = rsb_mtx_set_values(this->mtxAp,&rv,&ii,&jj,1,RSBOI_NF);
                                                            if(RSBOI_SOME_ERROR(errval))
							      error("FIXME: Incomplete: Can only accept already existing indices.");
							  }
							}
							RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
							retval = octave_value(this->clone());
						}
						else
						  error("FIXME: Incomplete: no complex sparse-sparse update for the moment.");
					}
#else /* RSBOI_WANT_SPMTX_SUBSASGN */
							RSBOI_DEBUG_NOTICE("UNFINISHED\n");
							idx_vector i = idx.front()(0).index_vector ();
							// ...
							RSBOI_IF_NERR(
								;//retval = octave_value (matrix.index (i, resize_ok));
							)
#endif /* RSBOI_WANT_SPMTX_SUBSASGN */
      						}
						break;
						default:
						{
							if (n_idx == 2 )
							{
								idx_vector i = idx.front() (0).index_vector ();
								idx_vector j = idx.front() (1).index_vector ();
#if 0
								// for op_el_div_eq and op_el_mul_eq
								std :: cout << "ic2 " << i.is_colon() << "\n" ;
								if( i.is_colon() && !j.is_colon() )
								{
									ComplexMatrix cm = rhs.complex_matrix_value();
									std :: cout << " : , .\n";
									errval=rsb_mtx_upd_values(this->mtxAp,RSB_ELOPF_SCALE_ROWS,cm.data());
								}
								if(!i.is_colon() &&  j.is_colon() )
								{
									std :: cout << " . , :\n";
								}
								if( i.is_colon() && j.is_colon() )
								{
									std :: cout << " : , :\n";
								}
#endif
								RSBOI_IF_NERR_STATE()
								{
									if(is_real_type())
									{
										rsb_err_t errval = RSB_ERR_NO_ERROR;
										rsb_coo_idx_t ii = -1, jj = -1;
										RSBOI_T rv = rhs.double_value();
										ii = i(0); jj = j(0);
										RSBOI_DEBUG_NOTICE("update elements (%d %d)\n",ii,jj);
#if RSBOI_WANT_SYMMETRY
										/* FIXME: and now ? */
#endif
										errval = rsb_mtx_set_values(this->mtxAp,&rv,&ii,&jj,1,RSBOI_NF);
										RSBOI_PERROR(errval);
										/* FIXME: I am unsure, here */
										//retval=rhs.double_value(); // this does not match octavej
										//retval=octave_value(this);
										retval = octave_value(this->clone()); // matches but .. heavy ?!
										RSBOI_IF_NERR(
											;//retval = octave_value (matrix.index (i, j, resize_ok));
										)
									}
									else
									{
										rsb_err_t errval = RSB_ERR_NO_ERROR;
										rsb_coo_idx_t ii = -1, jj = -1;
										Complex rv = rhs.complex_value();
										ii = i(0); jj = j(0);
										RSBOI_DEBUG_NOTICE("update elements (%d %d) complex\n",ii,jj);
#if RSBOI_WANT_SYMMETRY
				/* FIXME: and now ? */
#endif
										errval = rsb_mtx_set_values(this->mtxAp,&rv,&ii,&jj,1,RSBOI_NF);
										RSBOI_PERROR(errval);
										/* FIXME: I am unsure, here */
										//retval=rhs.double_value(); // this does not match octavej
										//retval=octave_value(this);
										retval = octave_value(this->clone()); // matches but .. heavy ?!
										RSBOI_IF_NERR(
											;//retval = octave_value (matrix.index (i, j, resize_ok));
										)
									}
//		  class octave_map;
//		  retval = octave_map();
//	RSBOI_DEBUG_NOTICE("UNFINISHED: set %d %d <- %lg\n",ii,jj,rhs.double_value());
	      							}
							}
						}
						break;
					}
					}
					else if (type.length () == 2)
					{
						std::list<octave_value_list>::const_iterator p =
							idx.begin ();
						octave_value_list key_idx = *++p;

						std::string key = key_idx(0).string_value ();
						RSBOI_DEBUG_NOTICE("UNFINISHED\n");

						if (key == "type")
							error ("use 'sparse_rsb' to set type");
						else
							error ("%s can indexed with .%s",
								type_name().c_str(), key.c_str());
					}
					else
						error ("in indexed assignment of %s, illegal assignment",
							type_name().c_str ());
				}
				break;
				case '.':
				{
					octave_value_list key_idx = idx.front ();
					std::string key = key_idx(0).string_value ();
					RSBOI_DEBUG_NOTICE("UNFINISHED\n");

					if (key == "type")
						error ("use 'sparse_rsb' to set matrix type");
					else
						error ("%s can indexed with .%s",
							type_name().c_str(), key.c_str());
				}
				break;

				case '{':
					RSBOI_DEBUG_NOTICE("UNFINISHED\n");
					error ("%s cannot be indexed with %c",
						type_name().c_str (), type[0]);
					break;

				default:
					panic_impossible ();
			}
			return retval;
		} /* subsasgn */

		octave_base_value *try_narrowing_conversion (void)
		{
			octave_base_value *retval = 0;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
			return retval;
		}

		/*
		type_conv_fcn numeric_conversion_function (void) const
		{
		}
		*/

		type_conv_info numeric_conversion_function (void) const
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			return default_numeric_conversion_function;
		}

		std::string get_info_string()
		{
			char ss[RSBOI_INFOBUF];
			rsb_mtx_get_info_str(this->mtxAp,"RSB_MIF_MATRIX_INFO__TO__CHAR_P",ss,RSBOI_INFOBUF);
			return ss;
		}

#if defined(OCTAVE_MAJOR_VERSION) && (OCTAVE_MAJOR_VERSION>=4)
		void print (std::ostream& os, bool pr_as_read_syntax = false)
#else  /* OCTAVE_MAJOR_VERSION */
		void print (std::ostream& os, bool pr_as_read_syntax = false) const
#endif /* OCTAVE_MAJOR_VERSION */
		{
			RSBOI_FIXME("what to do with pr_as_read_syntax ?");
			struct rsboi_coo_matrix_t rcm;
			rsb_err_t errval = RSB_ERR_NO_ERROR;
			rsb_nnz_idx_t nnzA = this->nnz(),nzi;
			bool ic = this->is_real_type()?false:true;
			Array<octave_idx_type> IA( dim_vector(1,nnzA) );
			Array<octave_idx_type> JA( dim_vector(1,nnzA) );
			Array<RSBOI_T> VA( dim_vector(1,(ic?2:1)*nnzA) );
			std::string c = ic ? "complex" : "real";
#if RSBOI_WANT_PRINT_DETAIL
			char ss[RSBOI_INFOBUF];
			rsb_mtx_get_info_str(this->mtxAp,"RSB_MIF_MATRIX_INFO__TO__CHAR_P",ss,RSBOI_INFOBUF);
#endif
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rcm.VA = (RSBOI_T*)VA.data(),rcm.IA = (octave_idx_type*)IA.data(),rcm.JA = (octave_idx_type*)JA.data();
#if RSBOI_WANT_SYMMETRY
			/* FIXME: and now ? */
#endif

			if(rcm.VA==RSBOI_NULL)
				nnzA = 0;
			else
				errval = rsboi_mtx_get_coo(this->mtxAp,rcm.VA,rcm.IA,rcm.JA,RSB_FLAG_C_INDICES_INTERFACE);
			rcm.nrA = this->rows();
			rcm.ncA = this->cols();
			double pct = 100.0*(((RSBOI_T)nnzA)/((RSBOI_T)rcm.nrA))/rcm.ncA;
			octave_stdout<<RSBOI_FSTR<< "  (rows = "<<rcm.nrA<<
				", cols = "<<rcm.ncA<<
				", nnz = "<<nnzA
#if RSBOI_WANT_SYMMETRY
				<< ", symm = "<<
				(RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_SYMMETRIC)?"S":
				(RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_SYMMETRIC)?"H":"U"))
				// FIXME: need a mechanism to print out these flags from rsb itself
#endif
			;
#if RSBOI_WANT_PRINT_PCT_OCTAVE_STYLE
			/* straight from Octave's src/ov-base-sparse.cc */
			if (nnzA > 0)
    			{
      				int prec = 2;
      				if (pct == 100) prec = 3; else { if (pct > 99.9) prec = 4; else if (pct > 99) prec = 3; if (pct > 99.99) pct = 99.99; }
      				octave_stdout << " [" << std::setprecision (prec) << pct << "%]";
    			}
#else
			octave_stdout << " ["<<pct<< "%]";
#endif

			octave_stdout <<
#if RSBOI_WANT_PRINT_COMPLEX_OR_REAL
				", "<<c<<
#endif
				")\n";
#if RSBOI_WANT_PRINT_DETAIL
			octave_stdout<< "{{"<< ss <<"}}\n";
#else
			octave_stdout<< "\n";
#endif
			if(ic)
			for(nzi=0;nzi<nnzA;++nzi)
				octave_stdout<<"  ("<<1+IA(nzi)<<", "<<1+JA(nzi)<<") -> "<<((RSBOI_T*)rcm.VA)[2*nzi+0]<<" + " <<((RSBOI_T*)rcm.VA)[2*nzi+1]<<"i\n";
			else
			for(nzi=0;nzi<nnzA;++nzi)
				octave_stdout<<"  ("<<1+IA(nzi)<<", "<<1+JA(nzi)<<") -> "<<((RSBOI_T*)rcm.VA)[nzi]<<"\n";
			newline(os);
			RSBIO_NULL_STATEMENT_FOR_COMPILER_HAPPINESS
		}

	octave_value diag (octave_idx_type k) const
	{
		octave_value retval;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		RSBOI_0_EMCHECK(this->mtxAp);

		if(k!=0)
		{
			error("only main diagonal extraction is supported !");
		}
		if(this->is_square())
		{
			rsb_err_t errval = RSB_ERR_NO_ERROR;
			//RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			if(this->is_real_type())
			{
				Matrix DA(this->rows(),1);
				errval = rsb_mtx_get_vec(this->mtxAp,(RSBOI_T*)DA.data(),RSB_EXTF_DIAG);
				retval = (DA);
			}
			else
			{
				ComplexMatrix DA(this->rows(),1);
				errval = rsb_mtx_get_vec(this->mtxAp,(RSBOI_T*)DA.data(),RSB_EXTF_DIAG);
				retval = (DA);
			}
		}
		else
		{
			error(RSBOI_0_NSQERRMSG);
		}
		return retval;
	}

	octave_value rsboi_get_scaled_copy_inv(const RSBOI_T alpha)const
	{
		RSBOI_T one = 1.0;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		return rsboi_get_scaled_copy(one/alpha);/* FIXME: is this correct ? */
	}

#if RSBOI_WANT_DOUBLE_COMPLEX
	octave_value rsboi_get_scaled_copy_inv(const Complex alpha)const
	{
		Complex one = 1.0;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		return rsboi_get_scaled_copy(one/alpha);/* FIXME: is this correct ? */
	}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

	octave_value rsboi_get_scaled_copy(const RSBOI_T alpha, rsb_trans_t transA=RSB_TRANSPOSITION_N)const
	{
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		struct rsb_mtx_t *mtxBp = RSBOI_NULL;

		if(is_real_type())
		{
			errval = rsb_mtx_clone(&mtxBp,RSB_NUMERICAL_TYPE_SAME_TYPE,transA, &alpha,this->mtxAp,RSBOI_EXPF);
		}
		else
#if RSBOI_WANT_DOUBLE_COMPLEX
		{
			Complex calpha;calpha+=alpha;
			errval = rsb_mtx_clone(&mtxBp,RSB_NUMERICAL_TYPE_SAME_TYPE,transA,&calpha,this->mtxAp,RSBOI_EXPF);
		}
#else /* RSBOI_WANT_DOUBLE_COMPLEX */
		{RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
		return new octave_sparsersb_mtx( mtxBp );
	}

#if RSBOI_WANT_DOUBLE_COMPLEX
	octave_value rsboi_get_scaled_copy(const Complex alpha)const
	{
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		octave_sparsersb_mtx *m = RSBOI_NULL;
		struct rsb_mtx_t *mtxBp = RSBOI_NULL;

		if(is_real_type())
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			errval = rsb_mtx_clone(&mtxBp,RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX,RSB_TRANSPOSITION_N,&rsboi_pone,this->mtxAp,RSBOI_EXPF);
			// FIXME: missing error handling!
			RSBOI_PERROR(errval);
			errval = rsb_mtx_upd_values(mtxBp,RSB_ELOPF_SCALE_ROWS,&alpha);
		}
		else
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			errval = rsb_mtx_clone(&mtxBp,RSB_NUMERICAL_TYPE_SAME_TYPE,RSB_TRANSPOSITION_N,&alpha,this->mtxAp,RSBOI_EXPF);
		}
		RSBOI_PERROR(errval);
		m = new octave_sparsersb_mtx( mtxBp );
		// FIXME: missing error handling!
		return m;
	}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

octave_value scale_rows(const octave_matrix&v2, bool want_div=false)
{
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

	if(this->is_real_type())
	{
		const Matrix rm = want_div?1.0/v2.matrix_value ():v2.matrix_value ();
		octave_idx_type b_nc = rm.cols ();
		octave_idx_type b_nr = rm.rows ();
		//octave_idx_type ldb = b_nr;
		octave_idx_type ldc = this->columns();
		octave_idx_type nrhs = b_nc;
		Matrix retval(ldc,nrhs,RSBOI_ZERO);
		if(this->rows()!=b_nr) { error("matrices dimensions do not match!\n"); return Matrix(); }
		errval = rsb_mtx_upd_values(this->mtxAp,RSB_ELOPF_SCALE_ROWS,rm.data());
		RSBOI_PERROR(errval);
		return retval;
	}
	else
	{
		const ComplexMatrix cm = want_div?1.0/v2.complex_matrix_value ():v2.complex_matrix_value ();
		octave_idx_type b_nc = cm.cols ();
		octave_idx_type b_nr = cm.rows ();
		//octave_idx_type ldb = b_nr;
		octave_idx_type ldc = this->columns();
		octave_idx_type nrhs = b_nc;
		ComplexMatrix retval(ldc,nrhs,RSBOI_ZERO);
		if(this->rows()!=b_nr) { error("matrices dimensions do not match!\n"); return ComplexMatrix(); }
		errval = rsb_mtx_upd_values(this->mtxAp,RSB_ELOPF_SCALE_ROWS,cm.data());
		RSBOI_PERROR(errval);
		return retval;
	}
}

octave_value rsboi_spmm(const octave_matrix&v2, bool do_trans=false)const
{
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	rsb_trans_t transA = do_trans ? RSB_TRANSPOSITION_T : RSB_TRANSPOSITION_N;

	if(this->is_real_type())
	{
		const Matrix b = v2.matrix_value ();
		octave_idx_type b_nc = b.cols ();
		octave_idx_type b_nr = b.rows ();
		octave_idx_type ldb = b_nr;
		octave_idx_type ldc = do_trans?this->columns():this->rows();
		octave_idx_type nrhs = b_nc;
		Matrix retval(ldc,nrhs,RSBOI_ZERO);

		// if(this->columns()!=b_nr) { error("matrices dimensions do not match!\n"); return Matrix(); }
		if(( do_trans)&&(this->rows()   !=b_nr)) { error("matrix rows count does not match operand rows!\n"); return Matrix(); }
		if((!do_trans)&&(this->columns()!=b_nr)) { error("matrix columns count does not match operand rows!\n"); return Matrix(); }
		errval = rsb_spmm(transA,&rsboi_pone,this->mtxAp,nrhs,RSB_OI_DMTXORDER,(RSBOI_T*)b.data(),ldb,&rsboi_zero,(RSBOI_T*)retval.data(),ldc);
		RSBOI_PERROR(errval);
		return retval;
	}
	else
	{
		const ComplexMatrix b = v2.complex_matrix_value ();
		octave_idx_type b_nc = b.cols ();
		octave_idx_type b_nr = b.rows ();
		octave_idx_type ldb = b_nr;
		octave_idx_type ldc = do_trans?this->columns():this->rows();
		octave_idx_type nrhs = b_nc;
		ComplexMatrix retval(ldc,nrhs,RSBOI_ZERO);

		if(( do_trans)&&(this->rows()   !=b_nr)) { error("matrix rows count does not match operand rows!\n"); return Matrix(); }
		if((!do_trans)&&(this->columns()!=b_nr)) { error("matrix columns count does not match operand rows!\n"); return Matrix(); }
		errval = rsb_spmm(transA,&rsboi_pone,this->mtxAp,nrhs,RSB_OI_DMTXORDER,(RSBOI_T*)b.data(),ldb,&rsboi_zero,(RSBOI_T*)retval.data(),ldc);
		RSBOI_PERROR(errval);
		return retval;
	}
}

#if RSBOI_WANT_DOUBLE_COMPLEX
octave_value rsboi_spmm(const octave_complex_matrix&v2, bool do_trans=false)const
{
	/*
		TODO: to avoid e.g. v2.complex_matrix_value, one may use: dim_vector  dv = v2.dims(); ... dv(ndims) ...
	*/
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	rsb_trans_t transA = do_trans == true ? RSB_TRANSPOSITION_T : RSB_TRANSPOSITION_N;
	struct rsb_mtx_t *mtxCp = RSBOI_NULL;
	const ComplexMatrix b = v2.complex_matrix_value ();
	octave_idx_type b_nc = b.cols ();
	octave_idx_type b_nr = b.rows ();
	octave_idx_type ldb = b_nr;
	octave_idx_type ldc = do_trans?this->columns():this->rows();
	octave_idx_type nrhs = b_nc;
	ComplexMatrix retval(ldc,nrhs,RSBOI_ZERO); /* zeroing is in principle unnecessary (we zero in rsb_spmm), but otherwise data may not be allocated. */
	RSBOI_T* Cp =(RSBOI_T*)retval.data();
	RSBOI_T* Bp =(RSBOI_T*)b.data();

	if(this->is_real_type())
		errval = rsb_mtx_clone(&mtxCp,RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX,RSB_TRANSPOSITION_N,RSBOI_NULL,this->mtxAp,RSBOI_EXPF);
	else
		mtxCp = this->mtxAp;
	if(RSBOI_SOME_ERROR(errval))
		goto err;

	if(( do_trans)&&(this->rows()   !=b_nr)) { error("matrix rows count does not match operand rows!\n"); return Matrix(); }
	if((!do_trans)&&(this->columns()!=b_nr)) { error("matrix columns count does not match operand rows!\n"); return Matrix(); }

	errval = rsb_spmm(transA,&rsboi_pone,mtxCp,nrhs,RSB_OI_DMTXORDER,Bp,ldb,&rsboi_zero,Cp,ldc);

	if(this->is_real_type())
		RSBOI_DESTROY(mtxCp);
err:
	RSBOI_PERROR(errval);
	return retval;
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

octave_value rsboi_spmsp(const octave_sparsersb_mtx&v2)const
{
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	octave_sparsersb_mtx*sm = new octave_sparsersb_mtx();
	octave_value retval = sm;

#if RSBOI_WANT_SYMMETRY
	/* FIXME: and now ? */
#endif
	/* FIXME: what if they are not both of the same type ? it would be nice to have a conversion.. */
	sm->mtxAp = rsb_spmsp(RSBOI_BINOP_PREVAILING_TYPE(*this,v2),RSB_TRANSPOSITION_N,&rsboi_pone,this->mtxAp,RSB_TRANSPOSITION_N,&rsboi_pone,v2.mtxAp,&errval);
	RSBOI_PERROR(errval);
	if(!sm->mtxAp)
		RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
	return retval;
}

octave_value rsboi_sppsp(const RSBOI_T*betap, const octave_sparsersb_mtx&v2)const
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	octave_sparsersb_mtx*sm = new octave_sparsersb_mtx();
	octave_value retval = sm;
	rsb_err_t errval = RSB_ERR_NO_ERROR;

	RSBOI_FIXME("");
#if RSBOI_WANT_SYMMETRY
	/* FIXME: and now ? */
#endif
	sm->mtxAp = rsb_sppsp(RSBOI_BINOP_PREVAILING_TYPE(*this,v2),RSB_TRANSPOSITION_N,&rsboi_pone,this->mtxAp,RSB_TRANSPOSITION_N,betap,v2.mtxAp,&errval);
	RSBOI_PERROR(errval);
	if(!sm->mtxAp)
		RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
	return retval;
}

#if RSBOI_WANT_DOUBLE_COMPLEX
octave_value cp_ubop(enum rsb_elopf_t opf, Complex z)const
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	octave_sparsersb_mtx *m = new octave_sparsersb_mtx(*this);

	if( is_real_type ())
	{
		struct rsb_mtx_t *mtxCp = RSBOI_NULL;
		errval = rsb_mtx_clone(&mtxCp,RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX,RSB_TRANSPOSITION_N,RSBOI_NULL,this->mtxAp,RSBOI_EXPF);
		if(RSBOI_SOME_ERROR(errval))
			goto err;
		errval = rsb_mtx_upd_values(mtxCp,opf,&z);
		// FIXME: need proper error handling
		RSBOI_PERROR(errval);
		RSBOI_DESTROY(m->mtxAp);
		m->mtxAp = mtxCp;
	}
	else
		errval = rsb_mtx_upd_values(m->mtxAp,opf,&z);
	// FIXME: need proper error handling
	RSBOI_PERROR(errval);
err:
	return m;
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

octave_value cp_ubop(enum rsb_elopf_t opf, void*alphap=RSBOI_NULL)const
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	octave_sparsersb_mtx *m = new octave_sparsersb_mtx(*this);

	if(!m)return m;
	errval = rsb_mtx_upd_values(m->mtxAp,opf,alphap);
	RSBOI_PERROR(errval);
	return m;
}

	private:
	public:
			DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};/* end of class octave_sparsersb_mtx definition  */

#if 0
octave_value_list find_nonzero_elem_idx (const class octave_sparsersb_mtx & nda, int nargout, octave_idx_type n_to_find, int direction)
{
	// useless
	octave_value retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return retval;
}
#endif

#if defined(RSBOI_USE_PATCH_38143)
#define RSBOI_CAST_CONV_ARG(ARGT) /* Seems like in 4.1.0+ CAST_CONV_ARG is not there. */	\
        ARGT v = dynamic_cast< ARGT > (a)
#define RSBOI_CAST_UNOP_ARG(ARGT) /* Seems like in 4.1.0+ CAST_UNOP_ARG is not there. */	\
	RSBOI_CAST_CONV_ARG(ARGT)
#define RSB_CAST_BINOP_ARGS(ARGT_V1, ARGT_V2); /* Seems like in 4.1.0+ CAST_BINOP_ARGS is not there. */	\
        ARGT_V1 v1 = dynamic_cast< ARGT_V1 > (a1);			\
        ARGT_V2 v2 = dynamic_cast< ARGT_V2 > (a2);
#else  /* RSBOI_USE_PATCH_38143 */
#define RSBOI_CAST_CONV_ARG CAST_CONV_ARG
#define RSBOI_CAST_UNOP_ARG CAST_UNOP_ARG
#define RSB_CAST_BINOP_ARGS CAST_BINOP_ARGS
#endif /* RSBOI_USE_PATCH_38143 */

static octave_base_value *default_numeric_conversion_function (const octave_base_value& a)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_CAST_CONV_ARG (const octave_sparsersb_mtx&);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_WARN(RSBOI_0_UNFFEMSG);
	if(v.is_real_type())
		return new octave_sparse_matrix (v.sparse_matrix_value());
	else
		return new octave_sparse_complex_matrix (v.sparse_complex_matrix_value());
}

DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA (octave_sparsersb_mtx,
RSB_OI_TYPEINFO_STRING,
RSB_OI_TYPEINFO_TYPE)

DEFCONV (octave_triangular_conv, octave_sparsersb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_CAST_CONV_ARG (const octave_sparsersb_mtx&);
	return new octave_sparse_matrix (v.matrix_value ());
}

#if 0
DEFCONV (octave_sparse_rsb_to_octave_sparse_conv, sparse_rsb_mtx, sparse_matrix)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_CAST_CONV_ARG (const octave_sparsersb_mtx&);
	return new octave_sparse_matrix (v.matrix_value ());
}
#endif

DEFUNOP (uplus, sparse_rsb_mtx)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	return new octave_sparsersb_mtx (v);
}

#if 0
DEFUNOP (op_incr, sparse_rsb_mtx)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	const octave_idx_type rn = v.mtxAp->nrA,cn = v.mtxAp->ncA;
	Matrix v2(rn,cn);
	octave_value retval = v2;
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	errval|=rsb_mtx_add_to_dense(&rsboi_pone,v.mtxAp,rn,rn,cn,RSB_BOOL_TRUE,(RSBOI_T*)v2.data());
	//v = octave_ma(idx, v2.matrix_value());
	return v2;
}

DEFUNOP (op_decr, sparse_rsb_mtx)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	const octave_idx_type rn = v.mtxAp->nrA, cn = v.mtxAp->ncA;
	Matrix v2(rn,cn);
	octave_value retval = v2;
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	errval|=rsb_mtx_add_to_dense(&rsboi_pone,v.mtxAp,rn,rn,cn,RSB_BOOL_TRUE,(RSBOI_T*)v2.data());
	//v = octave_ma(idx, v2.matrix_value());
	return v2;
}
#endif

DEFUNOP (uminus, sparse_rsb_mtx)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	return v.cp_ubop(RSB_ELOPF_NEG);
}

DEFUNOP (transpose, sparse_rsb_mtx)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	return v.rsboi_get_scaled_copy(rsboi_pone[0],RSB_TRANSPOSITION_T);
}

DEFUNOP (htranspose, sparse_rsb_mtx)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	return v.rsboi_get_scaled_copy(rsboi_pone[0],RSB_TRANSPOSITION_C);
}

octave_value rsboi_spsm(const octave_sparsersb_mtx&v1, const octave_matrix&v2, rsb_trans_t transA)
{
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

	if(v1.iscomplex())
	{
		ComplexMatrix retval = v2.complex_matrix_value();
		octave_idx_type b_nc = retval.cols ();
		octave_idx_type b_nr = retval.rows ();
		octave_idx_type ldb = b_nr;
		octave_idx_type ldc = v1.rows();
		octave_idx_type nrhs = b_nc;
		octave_idx_type nels = retval.rows()*retval.cols();
		errval = rsb_spsm(transA,&rsboi_pone,v1.mtxAp,nrhs,RSB_OI_DMTXORDER,&rsboi_zero,(const RSBOI_T*)retval.data(),ldb,(RSBOI_T*)retval.data(),ldc);
		if(RSBOI_SOME_ERROR(errval))
		{
			if(errval == RSB_ERR_INVALID_NUMERICAL_DATA)
			{
				RSBOI_PERROR(errval);// FIXME: need a specific error message here
			}
			else
			{
				RSBOI_PERROR(errval);// FIXME: generic case, here
			}
			for(octave_idx_type i=0;i<nels;++i)
				retval(i)=octave_NaN;
		}
		return retval;
	}
	else
	{
		Matrix retval = v2.matrix_value();
		octave_idx_type b_nc = retval.cols ();
		octave_idx_type b_nr = retval.rows ();
		octave_idx_type ldb = b_nr;
		octave_idx_type ldc = v1.rows();
		octave_idx_type nrhs = b_nc;
		octave_idx_type nels = retval.rows()*retval.cols();

		errval = rsb_spsm(transA,&rsboi_pone,v1.mtxAp,nrhs,RSB_OI_DMTXORDER,&rsboi_zero,(const RSBOI_T*)retval.data(),ldb,(RSBOI_T*)retval.data(),ldc);

		if(RSBOI_SOME_ERROR(errval))
		{
			if(errval == RSB_ERR_INVALID_NUMERICAL_DATA)
			{
				RSBOI_PERROR(errval);// FIXME: need a specific error message here
			}
			else
			{
				RSBOI_PERROR(errval);// FIXME: generic case, here
			}
			for(octave_idx_type i=0;i<nels;++i)
				retval(i)=octave_NaN;
		}
		return retval;
	}
}

#if RSBOI_WANT_DOUBLE_COMPLEX
octave_value rsboi_spsm(const octave_sparsersb_mtx&v1, const octave_complex_matrix&v2, rsb_trans_t transA)
{
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	ComplexMatrix retval = v2.complex_matrix_value();
	octave_idx_type b_nc = retval.cols ();
	octave_idx_type b_nr = retval.rows ();
	octave_idx_type ldb = b_nr;
	octave_idx_type ldc = v1.rows();
	octave_idx_type nrhs = b_nc;
	octave_idx_type nels = retval.rows()*retval.cols();
	struct rsb_mtx_t *mtxCp = RSBOI_NULL;

	if(v1.is_real_type())
		errval = rsb_mtx_clone(&mtxCp,RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX,RSB_TRANSPOSITION_N,RSBOI_NULL,v1.mtxAp,RSBOI_EXPF);
	else
		mtxCp = v1.mtxAp;
	if(RSBOI_SOME_ERROR(errval))
		goto err;

	errval = rsb_spsm(transA,&rsboi_pone,mtxCp,nrhs,RSB_OI_DMTXORDER,&rsboi_zero,(const RSBOI_T*)retval.data(),ldb,(RSBOI_T*)retval.data(),ldc);

	if(RSBOI_SOME_ERROR(errval))
	{
		if(errval==RSB_ERR_INVALID_NUMERICAL_DATA)
		{
			RSBOI_PERROR(errval);// FIXME: need a specific error message here
		}
		else
		{
			RSBOI_PERROR(errval);// FIXME: generic case, here
		}
		for(octave_idx_type i=0;i<nels;++i)
			retval(i)=octave_NaN;
	}
	if(v1.is_real_type())
		RSBOI_DESTROY(mtxCp);
err:
	RSBOI_PERROR(errval);
	return retval;
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

DEFBINOP(ldiv, sparse_rsb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_matrix&);

	if(v1.is__triangular())
		return rsboi_spsm(v1,v2,RSB_TRANSPOSITION_N);

	if(v1.iscomplex() || v2.iscomplex())
		return (v1.sparse_complex_matrix_value()).solve(v2.sparse_complex_matrix_value());
	else
		return (v1.sparse_matrix_value()).solve(v2.matrix_value());
	//RSBOI_RSB_MATRIX_SOLVE(v1,v2);
}

DEFBINOP(trans_ldiv, sparse_rsb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_matrix&);

	if(v1.is__triangular())
		return rsboi_spsm(v1,v2,RSB_TRANSPOSITION_T);

	if(v1.iscomplex() || v2.iscomplex())
		return (v1.sparse_complex_matrix_value().transpose()).solve(v2.sparse_complex_matrix_value());
	else
		return (v1.sparse_matrix_value().transpose()).solve(v2.matrix_value());
	//RSBOI_RSB_MATRIX_SOLVE(v1,v2);
}

#if RSBOI_WANT_DOUBLE_COMPLEX
DEFBINOP(c_ldiv, sparse_rsb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_complex_matrix&);

	if(v1.is__triangular())
		return rsboi_spsm(v1,v2,RSB_TRANSPOSITION_N);

	if(v1.iscomplex() || v2.iscomplex())
		return (v1.sparse_complex_matrix_value()).solve(v2.sparse_complex_matrix_value());
	else
		return (v1.sparse_matrix_value()).solve(v2.matrix_value());
	//RSBOI_RSB_MATRIX_SOLVE(v1,v2);
}

DEFBINOP(trans_c_ldiv, sparse_rsb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_complex_matrix&);
	if(v1.is__triangular())
		return rsboi_spsm(v1,v2,RSB_TRANSPOSITION_T);

	if(v1.iscomplex() || v2.iscomplex())
		return (v1.sparse_complex_matrix_value().transpose()).solve(v2.sparse_complex_matrix_value());
	else
		return (v1.sparse_matrix_value().transpose()).solve(v2.matrix_value());
	//RSBOI_RSB_MATRIX_SOLVE(v1,v2);
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

DEFBINOP(el_div, sparse_rsb_mtx, matrix)
{
	Matrix retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return retval;
}

DEFBINOP(el_ldiv, sparse_rsb_mtx, matrix)
{
	Matrix retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return retval;
}

DEFBINOP(div, sparse_rsb_mtx, matrix)
{
	Matrix retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return retval;
}

#if RSBOI_WANT_DOUBLE_COMPLEX
DEFBINOP(rsb_c_div, sparse_rsb_mtx, complex)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_complex&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.complex_value());
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

DEFBINOP(rsb_s_div, sparse_rsb_mtx, scalar)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.scalar_value());
}

DEFBINOP(rsb_s_mul, sparse_rsb_mtx, scalar)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy(v2.scalar_value());
}

DEFBINOP(s_rsb_mul, scalar, sparse_rsb_mtx)
{
	RSB_CAST_BINOP_ARGS (const octave_scalar&, const octave_sparsersb_mtx &);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v2.rsboi_get_scaled_copy(v1.scalar_value());
}

#if RSBOI_WANT_DOUBLE_COMPLEX
DEFBINOP(rsb_c_mul, sparse_rsb_mtx, complex)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_complex&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy(v2.complex_value());
}

DEFBINOP(c_rsb_mul, complex, sparse_rsb_mtx)
{
	RSB_CAST_BINOP_ARGS (const octave_complex&, const octave_sparsersb_mtx &);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v2.rsboi_get_scaled_copy(v1.complex_value());
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

#if RSBOI_WANT_POW
DEFBINOP(rsb_s_pow, sparse_rsb_mtx, scalar) // ^
{
	RSBOI_FIXME("This is elemental exponentiation!");
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_T alpha = v2.scalar_value();
	return v1.cp_ubop(RSB_ELOPF_POW,&alpha);
}
#endif /* RSBOI_WANT_POW */

DEFASSIGNOP (assign, sparse_rsb_mtx, sparse_rsb_mtx)
{
	rsb_err_t errval = RSB_ERR_NO_ERROR;
	RSBOI_FIXME("I dunno how to trigger this!");
	RSB_CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_sparsersb_mtx&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	//rsb_assign(v1.mtxAp, v2.mtxAp);
	errval = rsb_mtx_clone(&v1.mtxAp,RSB_NUMERICAL_TYPE_SAME_TYPE,RSB_TRANSPOSITION_N,RSBOI_NULL,v2.mtxAp,RSBOI_EXPF);
	return octave_value();
}

DEFASSIGNOP (assignm, sparse_rsb_mtx, matrix)
{
	RSB_CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_matrix&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_DESTROY(v1.mtxAp);
	octave_value retval;
	//v1.assign(idx, v2.matrix_value());
#if RSBOI_USE_PATCH_OCT44
	v1.assign(idx, v2.sparse_matrix_value());
#else /* RSBOI_USE_PATCH_OCT44 */
	// would break on octave6 (assignment deleted)
	v1 = (idx, v2.matrix_value());
#endif /* RSBOI_USE_PATCH_OCT44 */
	//retval = v1;
	retval = v2.matrix_value();
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return retval;
}

#if 0
DEFASSIGNOP(rsb_op_mul_eq_s, sparse_rsb_mtx, scalar)
{
	RSB_CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_scalar&);
	octave_value retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_PERROR(v1.rsboi_scale(v2.scalar_value()));
	retval = v1.matrix_value();
	return retval;
}

	rsb_err_t rsboi_scale(RSBOI_T alpha)
	{
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		//errval = rsb_elemental_scale(this->mtxAp,&alpha);
	       	errval = rsb_elemental_op(this->mtxAp,RSB_ELOPF_MUL,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

	rsb_err_t rsboi_scale(Complex alpha)
	{
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		//errval = rsb_elemental_scale(this->mtxAp,&alpha);
	       	errval = rsb_elemental_op(this->mtxAp,RSB_ELOPF_MUL,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

DEFASSIGNOP(rsb_op_div_eq_s, sparse_rsb_mtx, scalar)
{
	RSB_CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_scalar&);
	octave_value retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_PERROR(v1.rsboi_scale_inv(v2.scalar_value()));
	retval = v1.matrix_value();
	return retval;
}

	rsb_err_t rsboi_scale_inv(RSBOI_T alpha)
	{
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		//errval = rsb_elemental_scale_inv(this->mtxAp,&alpha);
	       	errval = rsb_elemental_op(this->mtxAp,RSB_ELOPF_DIV,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

	rsb_err_t rsboi_scale_inv(Complex alpha)
	{
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		//errval = rsb_elemental_scale_inv(this->mtxAp,&alpha);
	       	errval = rsb_elemental_op(this->mtxAp,RSB_ELOPF_DIV,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}
#endif

DEFBINOP(rsb_el_mul_s, sparse_rsb_mtx, scalar)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy(v2.scalar_value());
}

#if RSBOI_WANT_DOUBLE_COMPLEX
DEFBINOP(rsb_el_mul_c, sparse_rsb_mtx, complex)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_complex&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy(v2.complex_value());
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

DEFBINOP(rsb_el_div_s, sparse_rsb_mtx, scalar)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.scalar_value());
}

#if RSBOI_WANT_DOUBLE_COMPLEX
DEFBINOP(rsb_el_div_c, sparse_rsb_mtx, complex)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_complex&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.complex_value());
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

#if RSBOI_WANT_DOUBLE_COMPLEX
#if 0
DEFASSIGNOP(rsb_op_el_div_eq, sparse_rsb_mtx, scalar)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	std::cout << "rsb_op_el_div_eq!\n";
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.complex_value());
}
#endif

DEFASSIGNOP(rsb_op_el_mul_eq_sc, sparse_rsb_mtx, matrix)
{
	//rsb_err_t errval = RSB_ERR_NO_ERROR;
	RSB_CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_matrix&);
	return v1.scale_rows(v2,false);
}

DEFASSIGNOP(rsb_op_el_div_eq_sc, sparse_rsb_mtx, matrix)
{
	//rsb_err_t errval = RSB_ERR_NO_ERROR;
	RSB_CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_matrix&);
	return v1.scale_rows(v2,true);
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

DEFBINOP(el_pow, sparse_rsb_mtx, scalar)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_T alpha [] = {v2.scalar_value(),0};
	return v1.cp_ubop(RSB_ELOPF_POW,&alpha);
}

#if RSBOI_WANT_DOUBLE_COMPLEX
DEFBINOP(el_pow_c, sparse_rsb_mtx, complex)
{
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_complex&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	Complex alpha = v2.complex_value();
	return v1.cp_ubop(RSB_ELOPF_POW,alpha);
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

#ifdef RSB_FULLY_IMPLEMENTED
DEFASSIGNOP (assigns, sparse_rsb_mtx, scalar)
{
	RSB_CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	v1.assign(idx, v2.matrix_value());
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return octave_value();
}
#endif

DEFBINOP(op_sub, sparse_rsb_mtx, sparse_rsb_mtx)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_sparsersb_mtx&);
	return v1.rsboi_sppsp(&rsboi_mone[0],v2);
}

DEFBINOP(op_add, sparse_rsb_mtx, sparse_rsb_mtx)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_sparsersb_mtx&);
	return v1.rsboi_sppsp(&rsboi_pone[0],v2);
}

DEFBINOP(op_spmul, sparse_rsb_mtx, sparse_rsb_mtx)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_sparsersb_mtx&);
	return v1.rsboi_spmsp(v2);
}

DEFBINOP(op_mul, sparse_rsb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_matrix&);
	return v1.rsboi_spmm(v2, false);
}

DEFBINOP(op_trans_mul, sparse_rsb_mtx, matrix)
{
	// ".'*"  operator
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_matrix&);
	return v1.rsboi_spmm(v2, true);
}

#if RSBOI_WANT_DOUBLE_COMPLEX
DEFBINOP(op_c_mul, sparse_rsb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_complex_matrix&);
	return v1.rsboi_spmm(v2, false);
}

DEFBINOP(op_c_trans_mul, sparse_rsb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSB_CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_complex_matrix&);
	return v1.rsboi_spmm(v2, true);
}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

#if RSBOI_USE_PATCH_OCT44
#define RSBOI_INSTALL_BINOP(op, t1, t2, f) { \
  	octave::type_info& type_info = octave::__get_type_info__ ("");\
	type_info.register_binary_op(octave_value::op, t1::static_type_id (), t2::static_type_id (), CONCAT2 (oct_binop_, f)); }

#define RSBOI_INSTALL_ASSIGNOP(op, t1, t2, f) { \
  	octave::type_info& type_info = octave::__get_type_info__ ("");\
	type_info.register_assign_op(octave_value::op, t1::static_type_id (), t2::static_type_id (), CONCAT2 (oct_assignop_, f)); }

#define RSBOI_INSTALL_UNOP(op, t1, f) { \
  	octave::type_info& type_info = octave::__get_type_info__ ("");\
	type_info.register_unary_op(octave_value::op, t1::static_type_id (), CONCAT2 (oct_unop_, f)); }
#else /* RSBOI_USE_PATCH_OCT44 */
// deprecated; need a wrapper using octave::typeinfo::register_binary_op
#define RSBOI_INSTALL_BINOP INSTALL_BINOP

// deprecated; need a wrapper using octave::typeinfo::register_assign_op
#define RSBOI_INSTALL_ASSIGNOP INSTALL_ASSIGNOP

// deprecated; need a wrapper using octave::typeinfo::register_unary_op
#define RSBOI_INSTALL_UNOP INSTALL_UNOP
#endif /* RSBOI_USE_PATCH_OCT44 */

static void install_sparsersb_ops (void)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	#ifdef RSB_FULLY_IMPLEMENTED
	/* boolean pattern-based not */
	RSBOI_INSTALL_UNOP (op_not, octave_sparsersb_mtx, op_not);
	/* to-dense operations */
	RSBOI_INSTALL_ASSIGNOP (op_asn_eq, octave_sparsersb_mtx, octave_scalar, assigns);
	/* ? */
	RSBOI_INSTALL_UNOP (op_uplus, octave_sparsersb_mtx, uplus);
	/* elemental comparison, evaluate to sparse or dense boolean matrices */
	RSBOI_INSTALL_BINOP (op_eq, octave_sparsersb_mtx, , );
	RSBOI_INSTALL_BINOP (op_le, octave_sparsersb_mtx, , );
	RSBOI_INSTALL_BINOP (op_lt, octave_sparsersb_mtx, , );
	RSBOI_INSTALL_BINOP (op_ge, octave_sparsersb_mtx, , );
	RSBOI_INSTALL_BINOP (op_gt, octave_sparsersb_mtx, , );
	RSBOI_INSTALL_BINOP (op_ne, octave_sparsersb_mtx, , );
	/* pure elemental; scalar and sparse arguments ?! */
								 // ?
	RSBOI_INSTALL_BINOP (op_el_ldiv, octave_sparsersb_mtx, , );
	RSBOI_INSTALL_BINOP (op_el_ldiv_eq, octave_sparsersb_mtx, , ); // errval = rsb_mtx_upd_values(this->mtxAp,RSB_ELOPF_SCALE_ROWS,cm.data());
	RSBOI_INSTALL_BINOP (op_el_mul_eq, octave_sparsersb_mtx, , ); // diagonal subst ??
	RSBOI_INSTALL_BINOP (op_el_and, octave_sparsersb_mtx, , );
	RSBOI_INSTALL_BINOP (op_el_or, octave_sparsersb_mtx, , );
	/* shift operations: they may be left out from the implementation */
	RSBOI_INSTALL_BINOP (op_lshift, octave_sparsersb_mtx, , );
	RSBOI_INSTALL_BINOP (op_rshift, octave_sparsersb_mtx, , );
	#endif
	// RSBOI_INSTALL_ASSIGNOP (op_el_div_eq, octave_sparsersb_mtx, octave_matrix, rsb_op_el_div_eq_sc); // errval = rsb_mtx_upd_values(this->mtxAp,RSB_ELOPF_SCALE_ROWS,cm.data());
	// RSBOI_INSTALL_ASSIGNOP (op_el_mul_eq, octave_sparsersb_mtx, octave_matrix, rsb_op_el_mul_eq_sc);
	//INSTALL_WIDENOP (octave_sparsersb_mtx, octave_sparse_matrix,octave_sparse_rsb_to_octave_sparse_conv);/* a DEFCONV .. */
	//INSTALL_ASSIGNCONV (octave_sparsersb_mtx, octave_sparse_matrix,octave_sparse_matrix);/* .. */
	// no need for the following: need a good conversion function, though
	//RSBOI_INSTALL_UNOP (op_incr, octave_sparsersb_mtx, op_incr);
	//RSBOI_INSTALL_UNOP (op_decr, octave_sparsersb_mtx, op_decr);
	RSBOI_INSTALL_BINOP (op_el_mul, octave_sparsersb_mtx, octave_scalar, rsb_el_mul_s);
#if RSBOI_WANT_DOUBLE_COMPLEX
	RSBOI_INSTALL_BINOP (op_el_mul, octave_sparsersb_mtx, octave_complex, rsb_el_mul_c);
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
//	RSBOI_INSTALL_ASSIGNOP (op_mul_eq, octave_sparsersb_mtx, octave_scalar, rsb_op_mul_eq_s); // 20110313 not effective
//	RSBOI_INSTALL_ASSIGNOP (op_div_eq, octave_sparsersb_mtx, octave_scalar, rsb_op_div_eq_s); // 20110313 not effective
	RSBOI_INSTALL_BINOP (op_el_div, octave_sparsersb_mtx, octave_scalar, rsb_el_div_s);
#if RSBOI_WANT_DOUBLE_COMPLEX
	RSBOI_INSTALL_BINOP (op_el_div, octave_sparsersb_mtx, octave_complex, rsb_el_div_c);
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
	RSBOI_INSTALL_BINOP (op_el_pow, octave_sparsersb_mtx, octave_scalar, el_pow);
	RSBOI_INSTALL_BINOP (op_el_pow, octave_sparsersb_mtx, octave_complex, el_pow_c);
	RSBOI_INSTALL_UNOP (op_uminus, octave_sparsersb_mtx, uminus);
	RSBOI_INSTALL_BINOP (op_ldiv, octave_sparsersb_mtx, octave_matrix, ldiv);
	RSBOI_INSTALL_BINOP (op_el_ldiv, octave_sparsersb_mtx, octave_matrix, el_ldiv);
	RSBOI_INSTALL_BINOP (op_div, octave_sparsersb_mtx, octave_matrix, div);
	RSBOI_INSTALL_BINOP (op_div, octave_sparsersb_mtx, octave_scalar, rsb_s_div);
#if RSBOI_WANT_DOUBLE_COMPLEX
	RSBOI_INSTALL_BINOP (op_div, octave_sparsersb_mtx, octave_complex, rsb_c_div);
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
	RSBOI_INSTALL_BINOP (op_mul, octave_sparsersb_mtx, octave_scalar, rsb_s_mul);
	RSBOI_INSTALL_BINOP (op_mul, octave_scalar, octave_sparsersb_mtx, s_rsb_mul);
#if RSBOI_WANT_DOUBLE_COMPLEX
	RSBOI_INSTALL_BINOP (op_mul, octave_sparsersb_mtx, octave_complex, rsb_c_mul);
	RSBOI_INSTALL_BINOP (op_mul, octave_complex, octave_sparsersb_mtx, c_rsb_mul);
	RSBOI_INSTALL_BINOP (op_mul, octave_sparsersb_mtx, octave_complex_matrix, op_c_mul);
	RSBOI_INSTALL_BINOP (op_trans_mul, octave_sparsersb_mtx, octave_complex_matrix, op_c_trans_mul);
	RSBOI_INSTALL_BINOP (op_ldiv, octave_sparsersb_mtx, octave_complex_matrix, c_ldiv);
	RSBOI_INSTALL_BINOP (op_trans_ldiv, octave_sparsersb_mtx, octave_complex_matrix, trans_c_ldiv);
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
#if RSBOI_WANT_POW
	RSBOI_INSTALL_BINOP (op_pow, octave_sparsersb_mtx, octave_scalar, rsb_s_pow);
#endif /* RSBOI_WANT_POW */
	RSBOI_INSTALL_BINOP (op_el_div, octave_sparsersb_mtx, octave_matrix, el_div);
	RSBOI_INSTALL_UNOP (op_transpose, octave_sparsersb_mtx, transpose);
	RSBOI_INSTALL_UNOP (op_hermitian, octave_sparsersb_mtx, htranspose);
	RSBOI_INSTALL_ASSIGNOP (op_asn_eq, octave_sparsersb_mtx, octave_sparse_matrix, assign);
	RSBOI_INSTALL_ASSIGNOP (op_asn_eq, octave_sparsersb_mtx, octave_matrix, assignm);
	RSBOI_INSTALL_BINOP (op_mul, octave_sparsersb_mtx, octave_matrix, op_mul);
	//RSBOI_INSTALL_BINOP (op_pow, octave_sparsersb_mtx, octave_matrix, op_pow);
	RSBOI_INSTALL_BINOP (op_sub, octave_sparsersb_mtx, octave_sparsersb_mtx, op_sub);
	RSBOI_INSTALL_BINOP (op_add, octave_sparsersb_mtx, octave_sparsersb_mtx, op_add);
	//RSBOI_INSTALL_BINOP (op_trans_add, octave_sparsersb_mtx, octave_sparsersb_mtx, op_trans_add);
	RSBOI_INSTALL_BINOP (op_mul, octave_sparsersb_mtx, octave_sparsersb_mtx, op_spmul);
	RSBOI_INSTALL_BINOP (op_trans_mul, octave_sparsersb_mtx, octave_matrix, op_trans_mul);
	RSBOI_INSTALL_BINOP (op_trans_ldiv, octave_sparsersb_mtx, octave_matrix, trans_ldiv);
	//RSBOI_INSTALL_BINOP (op_mul_trans, octave_sparsersb_mtx, octave_matrix, op_mul_trans);
	//RSBOI_INSTALL_BINOP (op_mul_trans, octave_sparsersb_mtx, octave_matrix, op_mul_trans);
	//RSBOI_INSTALL_BINOP (op_herm_mul, octave_sparsersb_mtx, octave_matrix, op_herm_mul);
	//RSBOI_INSTALL_BINOP (op_mul_herm, octave_sparsersb_mtx, octave_matrix, op_mul_herm);
	//RSBOI_INSTALL_BINOP (op_el_not_and, octave_sparsersb_mtx, octave_matrix, op_el_not_and);
	//RSBOI_INSTALL_BINOP (op_el_not_or , octave_sparsersb_mtx, octave_matrix, op_el_not_or );
	//RSBOI_INSTALL_BINOP (op_el_and_not, octave_sparsersb_mtx, octave_matrix, op_el_and_not);
	//RSBOI_INSTALL_BINOP (op_el_or _not, octave_sparsersb_mtx, octave_matrix, op_el_or _not);
}

static void install_sparse_rsb (void)
{
	static bool rsboi_initialized = false;

	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

	if(!rsboi_initialized)
	{
		rsb_err_t errval = RSB_ERR_NO_ERROR;

		if(sparsersb_tester() == false)
		{
			RSBOI_ERROR("");
			goto err;
		}
		if(RSBOI_SOME_ERROR(errval = rsb_lib_init(RSB_NULL_INIT_OPTIONS)))
		{
			RSBOI_FIXME("temporary style of error handling");
			RSBOI_PERROR(errval);
			RSBOI_ERROR("");
			goto err;
		}
		rsboi_initialized = true;
	}
	else
		;/* already initialized */

	if (!rsboi_sparse_rsb_loaded)
	{
		octave_sparsersb_mtx::register_type ();
		install_sparsersb_ops ();
		rsboi_sparse_rsb_loaded = true;

#if RSBOI_USE_PATCH_OCT44
		octave::interpreter::the_interpreter()->mlock();
#else /* RSBOI_USE_PATCH_OCT44 */
		mlock();
#endif /* RSBOI_USE_PATCH_OCT44 */
	}
	return;
err:
	RSBIO_NULL_STATEMENT_FOR_COMPILER_HAPPINESS
} /* install_sparse_rsb */

DEFUN_DLD (RSB_SPARSERSB_LABEL, args, nargout,
"-*- texinfo -*-\n\
@deftypefn {Loadable Function} {@var{S} =} " RSBOI_FNS " (@var{a})\n\
@deftypefnx {Loadable Function} {@var{S} =} " RSBOI_FNS " (@var{i}, @var{j}, @var{sv}, @var{m}, @var{n})\n\
@deftypefnx {Loadable Function} {@var{S} =} " RSBOI_FNS " (@var{i}, @var{j}, @var{sv}, @var{m}, @var{n}, @var{nzmax})\n\
@deftypefnx {Loadable Function} {@var{S} =} " RSBOI_FNS " (@var{i}, @var{j}, @var{sv})\n\
@deftypefnx {Loadable Function} {@var{S} =} " RSBOI_FNS " (@var{m}, @var{n})\n\
@deftypefnx {Loadable Function} {@var{S} =} " RSBOI_FNS " (@var{i}, @var{j}, @var{sv}, @var{m}, @var{n}, \"unique\")\n\
@deftypefnx {Loadable Function}             " RSBOI_FNS " (\"set\", @var{opn}, @var{opv})\n\
@deftypefnx {Loadable Function} {@var{v} =} " RSBOI_FNS " (@var{S}, \"get\", @var{mif})\n\
@deftypefnx {Loadable Function} {@var{v} =} " RSBOI_FNS " (@var{S}, @var{QS})\n\
@deftypefnx {Loadable Function} " RSBOI_FNS " (@var{a},\"save\",@var{mtxfilename})\n\
@deftypefnx {Loadable Function} {[@var{S}, @var{nrows}, @var{ncols}, @var{nnz}, @var{repinfo}, @var{field}, @var{symmetry}] =} " RSBOI_FNS " (@var{mtxfilename}[, @var{mtxtypestring}])\n\
" RSBOI_10100_DOCH ""\
\
"\n"\
"Create or manipulate sparse matrices using the RSB format provided by librsb, as similarly as possible to @code{sparse}.\n"\
"\n"\
"If @var{a} is a full matrix, convert it to a sparse matrix representation,\n\
removing all zero values in the process.\n"\
"\n\
Given the integer index vectors @var{i} and @var{j}, and a 1-by-@code{nnz}\n\
vector of real or complex values @var{sv}, construct the sparse matrix\n\
@code{S(@var{i}(@var{k}),@var{j}(@var{k})) = @var{sv}(@var{k})} with overall\n\
dimensions @var{m} and @var{n}.  \n\
\nThe argument\n\
@code{@var{nzmax}} is ignored but accepted for compatibility with @sc{Matlab} and @code{sparsersb}.\n\
\n\
If @var{m} or @var{n} are not specified their values are derived from the\n\
maximum index in the vectors @var{i} and @var{j} as given by\n\
@code{@var{m} = max (@var{i})}, @code{@var{n} = max (@var{j})}.\n\
\n\
\
Can load a matrix from a Matrix Market matrix file named @var{mtxfilename}. The optional argument @var{mtxtypestring} can specify either real (\"D\") or complex (\"Z\") type. Default is real.\n"\
"In the case @var{mtxfilename} is \"" RSBOI_LIS "\", a string listing the available numerical types with BLAS-style characters will be returned. If the file turns out to contain a Matrix Market dense vector, this will be loaded.\n"\
\
\
"\n\
\
If \"save\" is specified, saves a sparse RSB matrix as a Matrix Market matrix file named @var{mtxfilename}.\n"\
"\n\
\
@strong{Note}: if multiple values are specified with the same\n\
@var{i}, @var{j} indices, the corresponding values in @var{sv} will\n\
be added.\n\
\n\
The following are all equivalent:\n\
\n\
@example\n\
@group\n\
s = " RSBOI_FNS " (i, j, s, m, n)\n\
s = " RSBOI_FNS " (i, j, s, m, n, \"summation\")\n\
s = " RSBOI_FNS " (i, j, s, m, n, \"sum\")\n"\
/*"s = " RSBOI_FNS " (i, j, s, \"summation\")\n"*/\
/*"s = " RSBOI_FNS " (i, j, s, \"sum\")\n"*/\
"@end group\n\
@end example\n\
\n\
\
If the optional \"unique\" keyword is specified, then if more than two values are specified for the\n\
same @var{i}, @var{j} indices, only the last value will be used.\n\
\n\
@code{" RSBOI_FNS " (@var{m}, @var{n})} will create an empty @var{m}x@var{n} sparse\n\
matrix and is equivalent to @code{" RSBOI_FNS " ([], [], [], @var{m}, @var{n})}.\n\
\n\
\
\n\
\
If @var{m} or @var{n} are not specified, then @code{@var{m} = max (@var{i})}, @code{@var{n} = max (@var{j})}.\n\
\n\
\
If @var{opn} is a string representing a valid librsb option name and @var{opv} is a string representing a valid librsb option value, these will be passed to the @code{rsb_lib_set_opt_str()} function.\n\
\n\
\
If @var{mif} is a string specifying a valid librsb matrix info string (valid for librsb's @code{rsb_mtx_get_info_from_string()}), then the corresponding value will be returned for matrix @code{@var{S}}, in string @code{@var{v}}. If @var{mif} is the an empty string (\"\"), matrix structure information will be returned. As of librsb-1.2, these is debug or internal information. E.g. for 'RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T', a string with the count of internal RSB blocks will be returned.\n\
\n"\
\
/*"If @var{S} is a " RSBOI_FNS " matrix and @var{QS} is a string, @var{QS} will be interpreted as a query string about matrix @var{S}. String @code{@var{v}} will be returned. See librsb's @code{rsb_mtx_get_info_str()}.\n\
@strong{Note}: this feature is still incomplete, and whatever the value of @var{QS}, a general information string will be returned.\n"*/\
\
"If @var{S} is a " RSBOI_FNS " matrix and @var{QS} is a string, @var{QS} shall be interpreted as a query string about matrix @var{S}. String @code{@var{v}} will be returned with query results. \n @strong{Note}: this feature is to be completed and its syntax reserved for future use. In this version, whatever the value of @var{QS}, a general matrix information string will be returned (like " RSBOI_FNS "(@var{S},\"get\",\"RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T\") ).\n"\
"\n"\
/*If any of @var{sv}, @var{i} or @var{j} are scalars, they are expanded\n\
to have a common size.\n*/
RSBOI_10100_DOC ""\
"\n\
Long (64 bit) index support is partial: if Octave has been configured for 64 bit indices, " RSBOI_FNS " will correctly handle and convert matrices/indices that would fit in a 32 bit indices setup, failing on 'larger' ones. \n\
\n\
Please note that on @code{" RSBOI_FNS "} type variables are available most, but not all of the operators available for @code{full} or @code{sparse} typed variables.\n\
\n\
@seealso{full, sparse}\n\
@end deftypefn")
{
	int nargin = args.length ();
	octave_value_list retval;
	octave_sparsersb_mtx*osmp = RSBOI_NULL;
	bool ic0 = nargin>0?(args(0).iscomplex()):false;
	bool ic3 = nargin>2?(args(2).iscomplex()):false;
	bool isr = (nargin>0 && args(0).type_name()==RSB_OI_TYPEINFO_STRING);

	RSBOI_DEBUG_NOTICE("in sparsersb()\n");

	if(ic0)
	{
		RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	}

	if(isr)
		osmp = ((octave_sparsersb_mtx*)(args(0).internal_rep()));

	if(ic3 || ic0)
#if RSBOI_WANT_DOUBLE_COMPLEX
		RSBOI_WARN(RSBOI_0_UNCFEMSG);
#else /* RSBOI_WANT_DOUBLE_COMPLEX */
		RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
	install_sparse_rsb();
	if( nargin == 3 && args(0).is_string() && args(0).string_value()=="set" && args(1).is_string() && args(2).is_string())
	{
		// sparsersb ("set", OPN, OPV)
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		const char *os = args(1).string_value().c_str();
		const char *ov = args(2).string_value().c_str();
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		errval = rsb_lib_set_opt_str(os,ov);
		if(RSBOI_SOME_ERROR(errval))
		{
			error("failed setting option %s to %s (error %d)!",os,ov,errval);
			goto err;
		}
		goto ret;
	}

	if( nargin >= 2 && args(0).is_string() && args(0).string_value()=="set" /* && args(1).is_string() */ )
	{
		// sparsersb ("set", XXX)
		error("did you intend to set librsb options ? use the correct syntax then ! (third argument missing)"); goto errp;
	}

	if( nargin == 2 && args(0).is_string() && args(0).string_value()=="get" && args(1).is_string() )
	{
		// sparsersb ("get", XXX)
		/* FIXME: unfinished feature ! */
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		error("getting library options still unimplemented!");
		goto errp;
	}

#if defined(RSB_LIBRSB_VER) && (RSB_LIBRSB_VER>=10100)
	if (nargin >= 2 && isr && args(1).is_string() && args(1).string_value()=="autotune")
	{
		// sparsersb (S,"autotune"[, TRANSA, NRHS, MAXR, TMAX, TN, SF])
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		/* these are user settable */
		rsb_coo_idx_t nrhs = 0;
		rsb_int_t maxr = 1;
		rsb_time_t tmax = 2.0;
		rsb_int_t tn = 0;
		rsb_real_t sf = 1.0;
		rsb_trans_t transA = RSB_TRANSPOSITION_N;
		/* TODO: these shall also be user settable */
		const void *alphap = RSBOI_NULL;
		const void *betap = RSBOI_NULL;
		/* these not */
	       	rsb_flags_t order = RSB_OI_DMTXORDER;
	       	const void * Bp = RSBOI_NULL;
		rsb_nnz_idx_t ldB = 0;
		rsb_nnz_idx_t ldC = 0;
		void * Cp = RSBOI_NULL;

		if (nargin > 2) transA = RSB_CHAR_AS_TRANSPOSITION(args(2).string_value()[0]);
		if (nargin > 3) nrhs = args(3).scalar_value();
		if (nargin > 4) maxr = args(4).scalar_value();
		if (nargin > 5) tmax = args(5).scalar_value();
		if (nargin > 6) tn = args(6).scalar_value();
		if (nargin > 7) sf = args(7).scalar_value();

		// ...
		if(!osmp || !osmp->mtxAp)
			RSBOI_0_INTERRMSGSTMT(goto ret)
		if(nargout)
		{
			struct rsb_mtx_t *mtxAp = RSBOI_NULL;
			errval = rsb_mtx_clone(&mtxAp,RSB_NUMERICAL_TYPE_SAME_TYPE,RSB_TRANSPOSITION_N,RSBOI_NULL,osmp->mtxAp,RSBOI_EXPF);
			errval = rsb_tune_spmm(&mtxAp,&sf,&tn,maxr,tmax,transA,alphap,RSBOI_NULL,nrhs,order,Bp,ldB,betap,Cp,ldC);
			retval.append(new octave_sparsersb_mtx(mtxAp));
		}
		else
			errval = rsb_tune_spmm(&osmp->mtxAp,&sf,&tn,maxr,tmax,transA,alphap,RSBOI_NULL/*osmp->mtxAp*/,nrhs,order,Bp,ldB,betap,Cp,ldC);
		/* FIXME: serious error handling missing here */
		goto ret;
	}
#endif


#if defined(RSB_LIBRSB_VER) && (RSB_LIBRSB_VER>=10100)
	if (nargin >= 3 && isr
 		&& args(1).is_string() && args(1).string_value().substr(0,6)=="render"
		&& args(2).is_string())
	{
		// sparsersb (S,"render", FILENAME[, RWIDTH, RHEIGHT])
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		std::string rmf = args(2).string_value();
		rsb_coo_idx_t pmWidth = 512, pmHeight = 512; /* Care to update the documentation when changing these. */
		rsb_flags_t marf = RSB_MARF_EPS;
		/* may tell the user to supply a sparsersb matrix in case input is not 'sparse' */

		if (nargin > 3) pmWidth = args(3).scalar_value();
		if (nargin > 4) pmHeight = args(4).scalar_value();

		if(!osmp || !osmp->mtxAp)
			RSBOI_0_INTERRMSGSTMT(goto ret)

 		if( args(1).string_value() == "renders")
			marf = RSB_MARF_EPS_S;
 		if( args(1).string_value() == "renderb")
			marf = RSB_MARF_EPS_B;
		errval = rsb_mtx_rndr(rmf.c_str(),osmp->mtxAp,pmWidth,pmHeight,marf);

		/* FIXME: serious error handling still missing here */
		if(RSBOI_SOME_ERROR(errval))
			retval.append(std::string("Error returned from rsb_mtx_rndr()"));
		goto ret;
	}
#endif
#if RSBOI_WANT_MTX_SAVE
	if (nargin == 3 && isr
 		&& args(1).is_string() && args(1).string_value()=="save"
		&& args(2).is_string())
	{
		// sparsersb (A,"save",MTXFILENAME)
		rsb_file_mtx_save(osmp->mtxAp,args(2).string_value().c_str()); /* TODO: error handling */
		goto ret;
	}
#endif
	if (nargin == 3 && isr
 		&& args(1).is_string() && args(1).string_value()=="get"
		&& args(2).is_string())
	{
		// sparsersb (S, "get", MIF)
		// For any version of lirsb, you can get valid values with e.g.:
		// grep RSB_MIF path-to/rsb.h | sed 's/^[, ]*//g;s/\([A-Z_]\+\).*<\(.\+\)(.*$/\1: \2/g;s/$/;/g'
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		/* rsb_real_t miv = RSBOI_ZERO;*/ /* FIXME: this is extreme danger! */
		char is[RSBOI_INFOBUF];
		char ss[RSBOI_INFOBUF];

		if(!osmp || !osmp->mtxAp)
			RSBOI_0_INTERRMSGSTMT(goto ret)

		if(strlen(args(2).string_value().c_str())==0)
			strncpy(is,"RSB_MIF_MATRIX_INFO__TO__CHAR_P",sizeof(is));
		else
			strncpy(is,args(2).string_value().c_str(),sizeof(is));
		errval = rsb_mtx_get_info_str(osmp->mtxAp,is,ss,RSBOI_INFOBUF);

		if(!RSBOI_SOME_ERROR(errval))
		{
			retval.append(octave_value(ss));
			goto ret;
		}
		/* FIXME: serious error handling missing here */
		if(RSBOI_SOME_ERROR(errval))
			retval.append(std::string("Error returned from rsb_mtx_get_info_from_string()"));
	/*	else
			retval.append(octave_value(miv));*/
		goto ret;
	}

	if ( nargin >= 3 && isr && args(1).is_string() && args(1).string_value()=="get" /* && args(1).is_string() */ )
	{
		// sparsersb (S, "get", MIF, XXX)
		error("did you intend to get matrices information ? use the correct syntax then !");
		goto errp;
	}

	if ( nargin == 1 || nargin == 2 )
	{
		rsb_type_t typecode = RSBOI_TYPECODE;
		if (nargin >= 2)/* FIXME: this is weird ! */
#if RSBOI_WANT_DOUBLE_COMPLEX
			typecode = RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX;
#else /* RSBOI_WANT_DOUBLE_COMPLEX */
			RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */

		if (nargin == 2 && isr && args(1).is_string())
#if RSBOI_WANT_QSI
		{
			// sparsersb (S, QS)
			char ss[RSBOI_INFOBUF];
			rsb_err_t errval = RSB_ERR_NO_ERROR;

			if(!osmp || !osmp->mtxAp)
				RSBOI_0_INTERRMSGSTMT(goto ret)
			errval = rsb_mtx_get_info_str(osmp->mtxAp,"RSB_MIF_MATRIX_INFO__TO__CHAR_P",ss,RSBOI_INFOBUF);
			if(!RSBOI_SOME_ERROR(errval))
				retval.append(ss);
			/* TODO, FIXME: to add interpretation (we are ignoring args(1) !): this is to be extended. */
			RSBOI_WARN(RSBOI_0_UNFFEMSG);/* FIXME: this is yet unfinished */
			// octave_stdout << "Matrix information (in the future, supplementary information may be returned, as more inquiry functionality will be implemented):\n" << ss << "\n";
			/* FIXME: shall not print out, but rather return the info as a string*/
			//retval.append("place info string here !\n");
			goto ret;
		}
#else /* RSBOI_WANT_QSI */
		{
			// sparsersb (S, QS)
			error("invocation error !");
		       	goto errp;
		}
#endif /* RSBOI_WANT_QSI */
		else
		if(args(0).issparse())
		{
			// sparsersb (sparse(...), ...)
			if( isr )
			{
				RSBOI_WARN(RSBOI_0_UNFFEMSG);
				retval.append(osmp = (octave_sparsersb_mtx*)(args(0).get_rep()).clone());
			}
			else
			{
				if(!ic0)
				{
					const SparseMatrix m = args(0).sparse_matrix_value();
					RSBOI_IF_ERR( goto err;)
					retval.append(osmp = new octave_sparsersb_mtx(m,typecode));
				}
#if RSBOI_WANT_DOUBLE_COMPLEX
				else
				{
					const SparseComplexMatrix m = args(0).sparse_complex_matrix_value();
					RSBOI_IF_ERR( goto err;)
					retval.append(osmp = new octave_sparsersb_mtx(m,typecode));
				}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
			}
		}
		else
		if(args(0).is_string())
		{
		RSBOI_TRY_BLK
		{
			// sparsersb (MTXFILENAME)
			const std::string mtxfilename = args(0).string_value();
			RSBOI_IF_ERR( goto err;)
			if(mtxfilename == RSBOI_LIS)
			{
				//retval.append(RSB_NUMERICAL_TYPE_PREPROCESSOR_SYMBOLS);
#if RSBOI_WANT_DOUBLE_COMPLEX
				retval.append("D Z");
#else /* RSBOI_WANT_DOUBLE_COMPLEX */
				retval.append("D");
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
				goto ret;
			}
			else
			{
				// [S, NROWS, NCOLS, NNZ, REPINFO, FIELD, SYMMETRY] = sparsersb (MTXFILENAME)
				rsb_type_t typecode = RSBOI_TYPECODE;
				RSBOI_WARN(RSBOI_0_UNFFEMSG);
				RSBOI_WARN("shall set the type, here");
				if(nargin>1 && args(1).is_string())
				{
					const std::string mtxtypestring = args(1).string_value();
					if(mtxtypestring == "complex" || mtxtypestring == "Z")
#if RSBOI_WANT_DOUBLE_COMPLEX
						typecode = RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX;
#else
						RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
					if(mtxtypestring == "real" || mtxtypestring=="D")
						typecode = RSB_NUMERICAL_TYPE_DOUBLE;
				}
#if RSBOI_WANT_MTX_LOAD
				osmp = new octave_sparsersb_mtx(mtxfilename,typecode);
#else /* RSBOI_WANT_DOUBLE_COMPLEX */
				goto ret; /* TODO: need error message here */
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
				if(osmp->mtxAp)
					retval.append(osmp);
				else
					delete osmp;
#if RSBOI_WANT_VECLOAD_INSTEAD_MTX
				if(!osmp->mtxAp)
                		{
					rsb_nnz_idx_t n = 0;
					rsb_file_vec_load(mtxfilename.c_str(),typecode,RSBOI_NULL,&n);
					if(n<1)
					{
						// error("are you sure you passed a valid Matrix Market vector file ?");
						goto err;
					}

					if(typecode == RSB_NUMERICAL_TYPE_DOUBLE)
					{
						Matrix retvec(n,1,RSBOI_ZERO);
						rsb_file_vec_load(mtxfilename.c_str(),typecode,(RSBOI_T*)retvec.data(),&n);
						retval.append(retvec);
					}
#if RSBOI_WANT_DOUBLE_COMPLEX
					else
					if(typecode == RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX)
					{
						ComplexMatrix retvec(n,1,RSBOI_ZERO);
						rsb_file_vec_load(mtxfilename.c_str(),typecode,(RSBOI_T*)retvec.data(),&n);
						retval.append(retvec);
					}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
					goto ret;
				}
#endif
				if(nargout) nargout--;
				if(nargout) retval.append(osmp->rows()),--nargout;
				if(nargout) retval.append(osmp->cols()),--nargout;
				if(nargout) retval.append(osmp->nnz()),--nargout;
				if(nargout) retval.append(osmp->get_info_string()),--nargout;
				if(nargout) retval.append((!osmp->iscomplex())?"real":"complex"),--nargout;
				if(nargout) retval.append(osmp->get_symmetry()),--nargout;
			}
		}
		RSBOI_CATCH_BLK
		}
		else
		{
		RSBOI_TRY_BLK
		{
			if (nargin == 2  && args(0).is_scalar_type() && args(1).is_scalar_type() )
			{
				// sparsersb (M, N)
				const SparseMatrix m = args(0).sparse_matrix_value();
				retval.append(osmp = new octave_sparsersb_mtx(SparseMatrix(args(0).scalar_value(),args(1).scalar_value())));
			}
			else
			{
				// sparsersb (A, XXX)
				if(!ic0)
				{
					Matrix m = args(0).matrix_value();
					RSBOI_IF_ERR( goto err;)
					retval.append(osmp = new octave_sparsersb_mtx(m));
				}
#if RSBOI_WANT_DOUBLE_COMPLEX
				else
				{
					ComplexMatrix m = args(0).complex_matrix_value();
					RSBOI_IF_ERR( goto err;)
					retval.append(osmp = new octave_sparsersb_mtx(m));
				}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
				if(nargin >= 2)
				{ error("when initializing from a single matrix, no need for second argument !"); goto errp; }
			}
		}
		RSBOI_CATCH_BLK
		}
	}
	else
	if (nargin >= 3 && nargin <= 7 && !(args(0).is_string() || args(1).is_string() || args(2).is_string() ) )
	{
		// sparsersb (I, J, SV, M, N, "unique")
		rsb_flags_t eflags = RSBOI_DCF;
		rsb_flags_t sflags = RSB_FLAG_NOFLAGS;
		octave_idx_type nrA = 0, ncA = 0;
		int sai = 0; // string argument index

		if (nargin > 3)
		{
			if ( nargin < 5)
			{
				if(nargin == 4 && args(3).is_string())
					goto checked;
				RSBOI_EERROR(RSBOI_0_BADINVOERRMSG);
				goto errp;
			}
			/* FIXME: integer_type should be also supported here: shouldn't it ?*/
    			if( (!args(3).is_scalar_type()) || (!args(4).is_scalar_type()))
			{
				RSBOI_EERROR(RSBOI_0_BADINVOERRMSG);
				goto errp;
			}
     			if( nargin > 5 && ((!args(5).is_string()) && (!args(5).is_scalar_type())))
			{
				RSBOI_EERROR(RSBOI_0_BADINVOERRMSG);
				goto errp;
			}
		}
checked:
		if (nargin >= 5  )
		{
			nrA = args(3).scalar_value();/* FIXME: need index value here! */
			ncA = args(4).scalar_value();
			if(nrA<=0 || ncA<=0)
			{
				RSBOI_EERROR(RSBOI_O_NPMSERR);
				goto errp;
			}
		}

		if (nargin >= 6  && args(5).is_string())
			sai = 5;
		else
			if (nargin == 4  && args(3).is_string())
				sai = 3;
		for(;sai>0 && sai<nargin;++sai)
		{
			std::string vv = args(sai).string_value();

			if ( vv == "summation" || vv == "sum" )
				eflags = RSB_FLAG_DUPLICATES_SUM;
			else
			if ( vv == "unique" )
				eflags = RSB_FLAG_DUPLICATES_KEEP_LAST;
#if RSBOI_WANT_SYMMETRY
			/* FIXME: still undocumented extension */
			else
			if ( vv == "symmetric" || vv == "sym" )
				sflags = RSB_FLAG_SYMMETRIC;
			else
			if ( vv == "hermitian" || vv == "her" )
				sflags = RSB_FLAG_HERMITIAN;
			else
			if ( vv == "general" || vv == "gen" )
				;
#endif /* RSBOI_WANT_SYMMETRY */
			else
			{
				vv = "'" + vv;
				vv+="' is not a recognized keyword (unlike 'summation', 'unique', 'symmetric', 'hermitian', 'general')!";
				error("%s",vv.c_str());
				goto errp;
			}
		}
		RSB_DO_FLAG_ADD(eflags,sflags);
		if (nargin >= 6  && args(5).isinteger())
		{
			/* we ignore this value for MATLAB compatibility */
		}

		RSBOI_IF_ERR( goto err;)

		if(!ic3)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			idx_vector iv = args(0).index_vector ();
			idx_vector jv = args(1).index_vector ();
			retval.append(osmp = new octave_sparsersb_mtx( iv, jv, args(2).matrix_value(),nrA,ncA,eflags ));
		}
#if RSBOI_WANT_DOUBLE_COMPLEX
		else
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			idx_vector iv = args(0).index_vector ();
			idx_vector jv = args(1).index_vector ();
			retval.append(osmp = new octave_sparsersb_mtx( iv, jv, args(2).complex_matrix_value(),nrA,ncA,eflags ));
		}
#endif /* RSBOI_WANT_DOUBLE_COMPLEX */
	}
	else
		goto errp;
	if(!osmp)
	{
		RSBOI_WARN(RSBOI_0_NEEDERR);
		RSBOI_DEBUG_NOTICE(RSBOI_0_FATALNBMSG);
	}
#if RSBOI_WANT_HEAVY_DEBUG
	if(!rsb_is_correctly_built_rcsr_matrix(osmp->mtxAp)) // function non in rsb.h's API
	{
		RSBOI_WARN(RSBOI_0_NEEDERR);
		RSBOI_DEBUG_NOTICE(RSBOI_0_UNCBERR);
	}
#endif
	goto err;
	errp:
	print_usage ();
err:
ret:
	return retval;
}
/*
%!test
%! #help sparsersb
%! s=sparsersb([2]), assert(s==2), assert(s!=1)
%!test
%! s=sparsersb([1,2],[1,1],[11,21],2,2         ), assert(nnz(s)==2)
%!test
%! s=sparsersb([1,2],[1,1],[11,21],2,2,-1      ), assert(nnz(s)==2)
%!test
%! s=sparsersb([1,2],[1,1],[11,21]             ), assert(nnz(s)==2)
%!test
%! s=sparsersb(10,10                           ), assert(nnz(s)==0)
%!test
%! s=sparsersb([1,1],[1,1],[11,21]             ), assert(nnz(s)==1), assert(s(1,1)==32)
%!test
%! s=sparsersb([1,1],[1,1],[11,21],2,2,"unique"), assert(nnz(s)==1), assert(s(1,1)==21)
%!test
%! sparsersb("set","RSB_IO_WANT_VERBOSE_TUNING","1")
%!test
%! # sparsersb("get","RSB_IO_WANT_VERBOSE_TUNING","1") # FIXME
%!test
%! sparsersb(sparsersb([11,0;21,22]),"RSB_MIF_TOTAL_SIZE__TO__SIZE_T")
%!test
%! sparsersb(sparsersb([11,0;21,22]),"save","sparsersb_temporary_matrix_file.mtx")
%!test
%! [S, NROWS, NCOLS, NNZ, REPINFO, FIELD, SYMMETRY] = sparsersb("sparsersb_temporary_matrix_file.mtx"     ); assert(NROWS==2);assert(NCOLS==2);assert(NNZ==3);assert(FIELD=="real"   );assert(SYMMETRY=='U');
%!test
%! [S, NROWS, NCOLS, NNZ, REPINFO, FIELD, SYMMETRY] = sparsersb("sparsersb_temporary_matrix_file.mtx", "Z"); assert(NROWS==2);assert(NCOLS==2);assert(NNZ==3);assert(FIELD=="complex");assert(SYMMETRY=='U');
%!test
%! rrm=sparsersb(sprand(1000,1000,0.001)); sparsersb(rrm,"render", "sparsersb_temporary_render.eps" ,1024,1024);
%! # sparsersb(rrm,"renderb", "sparsersb_temporary_renderb.eps"); sparsersb(rrm,"renders", "sparsersb_temporary_renders.eps"); # FIXME
%!test
%! sparsersb(sparsersb(sprand(100,100,0.4)),"autotune","n",20,4,1,1,1)
*/

/*
%!demo
%! # You can use 'sparsersb' just like 'sparse' in the most of cases:
%! R=(rand(3)>.6)
%! # R =
%! #
%! #    0   0   0
%! #    0   0   0
%! #    1   0   1
%! #
%! A_octave=sparse(R)
%! # A_octave =
%! #
%! # Compressed Column Sparse (rows = 3, cols = 3, nnz = 2 [22%])
%! #
%! #   (3, 1) ->  1
%! #   (3, 3) ->  1
%! #
%! A_librsb=sparsersb(R)
%! # A_librsb =
%! #
%! # Recursive Sparse Blocks  (rows = 3, cols = 3, nnz = 2, symm = U [22%])
%! #
%! #   (3, 1) -> 1
%! #   (3, 3) -> 1
%! #
%! # test sparsersb
%! # ...
%! # help sparsersb

%!demo
%! # The interface of 'sparsersb' is almost like the one of 'sparse'.
%! sparsersb([2]); # 1x1 matrix
%! sparsersb([1,2],[1,1],[11,21]    ); # 2x1 matrix
%! sparsersb([1,2],[1,1],[11,21],2,2); # 2x2 matrix
%! sparsersb([1,2,2  ],[1,1,2  ],[11,21,   22],2,2);          # 2x2 lower triangular
%! sparsersb([1,2,2,2],[1,1,2,2],[11,21,11,11],2,2);          # 2x2 lower triangular, last element ignored
%! sparsersb([1,2,2,2],[1,1,2,2],[11,21,11,11],2,2,"unique"); # 2x2 lower triangular, last element ignored
%! sparsersb([1,2,2,2],[1,1,2,2],[11,21,11,11],2,2,"sum");    # 2x2 lower triangular, last two elements summed
%!
%! # But it has a extensions, like symmetric and hermitian matrices.
%! sparsersb([1,2,2  ],[1,1,2  ],[11,21 ,  22],2,2,"general");   # 2x2 lower tringular
%! sparsersb([1,2,2  ],[1,1,2  ],[11,21 ,  22],2,2,"symmetric"); # 2x2 symmetric (only lower triangle stored)
%! sparsersb([1,2,2  ],[1,1,2  ],[11,21i,  22],2,2,"hermitian"); # 2x2 hermitian (only lower triangle stored)

%!demo
%! # Any 'sparse' or 'dense' matrix can be converted to 'sparsersb'.
%! d=sparsersb(       [1,2;3,4] );
%! s=sparsersb(sparse([1,2;3,4]));
%!
%! # Many matrix operators are active, e.g.: +,*,-,/,\ among others...
%! s+d;
%! s*d;
%! s-d;
%! s/d;
%! s\[1;1];
%! # ...

%!demo
%! # On large matrices 'sparsersb' is supposed to be faster than 'sparse' in sparse matrix-vector multiplication:
%! M=10000;N=10000;P=100 / M;
%! s=sparse(sprand(M,N,P));
%! r=sparsersb(s);
%! x=ones(M,1);
%! assert(nnz(s)==nnz(r))
%!
%! printf("Here, a %.2e x %.2e matrix with %.2e nonzeroes.\n",M,N,nnz(s))
%! tic();
%! sc=0;
%! while(toc()<3)
%!   s*x;
%!   sc=sc+1;
%! endwhile
%! st=toc()/sc;
%! printf("Each multiplication with 'sparse' took %.1es.\n",st);
%!
%! tic();
%! rc=0;
%! while(toc()<3)
%!   r*x;
%!   rc=rc+1;
%! endwhile
%! rt=toc()/rc;
%! printf("Each multiplication with 'sparsersb' took %.3es, this is %.4g%% of the time taken by 'sparse'.\n",rt,100*rt/st);
%!
%! # 'sparsersb' has an 'empirical online auto-tuning' function
%! nsb=str2num(sparsersb(r,"get","RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T"));
%! # after 'autotuning' for a specific operation, this will perform faster
%! tic;
%! r=sparsersb(r,"autotune","n",1);
%! toc;
%! nnb=str2num(sparsersb(r,"get","RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T"));
%! printf ("Autotuning took  %.2es (%d -> %d RSB blocks).\n", toc, nsb, nnb);
%! tic();
%! rc=0;
%! while(toc()<3)
%!   r*x;
%!   rc=rc+1;
%! endwhile
%! rt=toc()/rc;
%! printf("Each 'optimized' multiplication with 'sparsersb' took %.3es, this is %.4g%% of the time taken by 'sparse'.\n",rt,100*rt/st);

%!demo
%! # 'sparsersb' can render matrices into Encapsulated Postscript files:
%! rm = sparsersb(sprand(100000,100000,.0001));
%! sparsersb(rm,'render','sptest.eps')
%%!demo
*/

