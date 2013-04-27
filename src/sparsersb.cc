/*
 Copyright (C) 2011-2013   Michele Martone   <michelemartone _AT_ users.sourceforge.net>

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
 * TODO wishlist:
 * ("get","RSB_IO_WANT_...") is not yet available
 * (.) is incomplete. it is needed by trace()
 * (:,:) , (:,p) ... do not work, test with octave's bicg, bicgstab, cgs, ...
 * hints about how to influence caching blocking policy
 * compound_binary_op
 * for thorough testing, see Octave's test/build_sparse_tests.sh
 * introspection functionality (bytes/nnz, or  sparsersb(rsbmat,"inquire: subm") )
 * sparsersb(rsbmat,"autotune")
 * sparsersb(rsbmat,"benchmark")
 * sparsersb(rsbmat,"test")
 * minimize data copies
 * subsref, dotref, subsasgn are incomplete: need error messages there
 * in full_value(), bool arg is ignored
 * symmetry support is incomplete
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

 * Developer notes:
 http://www.gnu.org/software/octave/doc/interpreter/index.html
 http://www.gnu.org/software/octave/doc/interpreter/Oct_002dFiles.html#Oct_002dFiles
 http://octave.sourceforge.net/developers.html
 */

#define RSBOI_WANT_PRINT_PCT_OCTAVE_STYLE 1

#include <octave/oct.h>
#include <octave/ov-re-mat.h>
#include <octave/ov-re-sparse.h>
#include <octave/ov-scalar.h>
#include <octave/ov-complex.h>
#include <octave/ops.h>
#include <octave/ov-typeinfo.h>
#if RSBOI_WANT_PRINT_PCT_OCTAVE_STYLE
#include <iomanip>	// std::setprecision
#endif
#include <rsb.h>

#ifdef RSBOI_VERBOSE_CONFIG
#if (RSBOI_VERBOSE_CONFIG>0)
#define RSBOI_VERBOSE RSBOI_VERBOSE_CONFIG
#endif
#endif

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
//#define RSBOI_ERROR( ... ) \
	printf("In %s(), in file %s at line %10d:\n",__func__,__FILE__,__LINE__), \
	printf( __VA_ARGS__ )
#define RSBOI_ERROR( MSG ) \
	octave_stdout << "In "<<__func__<<"(), in file "<<__FILE__<<" at line "<<__LINE__<<":\n"<<MSG
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
#define RSBOI_DESTROY(OM) {rsb_mtx_free(OM);(OM)=NULL;}
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
#define RSBOI_D_EMPTY_MSG  ""
#define RSBOI_O_MISSIMPERRMSG  "implementation missing here\n"
#define RSBOI_O_NPMSERR  "providing non positive matrix size is not allowed!"
#define RSBOI_0_EMCHECK(M) if(!(M))RSBOI_0_ERROR(RSBOI_0_EMERRMSG);
#define RSBOI_FNSS(S)	#S
//#define RSBOI_FNS	RSBOI_FNSS(RSB_SPARSERSB_LABEL)
#define RSBOI_FSTR	"Recursive Sparse Blocks"
#define RSBOI_FNS	"sparsersb"
#define RSBOI_LIS	"?"

#define RSBIO_DEFAULT_CORE_MATRIX  Matrix (0,0)
/* FIXME : octave_idx_type vs rsb_coo_idx_t */
#define RSBIO_NULL_STATEMENT_FOR_COMPILER_HAPPINESS {1;}
#define RSBOI_OV_STRIDE 1
#define RSBOI_ZERO 0.0
//#define RSB_OI_DMTXORDER RSB_FLAG_WANT_ROW_MAJOR_ORDER 
#define RSB_OI_DMTXORDER RSB_FLAG_WANT_COLUMN_MAJOR_ORDER  /* for dense matrices (multivectors) */ 
#define RSB_OI_TYPEINFO_STRING "rsb sparse matrix"
#define RSB_OI_TYPEINFO_TYPE    "double"

#ifdef RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX
#define RSBOI_WANT_DOUBLE_COMPLEX 1
#define ORSB_RSB_TYPE_FLAG(OBJ) (((OBJ).is_complex_type())?RSB_NUMERICAL_TYPE_DOUBLE:RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX)
#else
#define RSBOI_WANT_DOUBLE_COMPLEX 0
#define ORSB_RSB_TYPE_FLAG(OBJ) RSB_NUMERICAL_TYPE_DOUBLE
#endif

#define RSBOI_INFOBUF	256
#define RSBOI_WANT_SYMMETRY 0
#define RSBOI_WANT_PRINT_DETAIL 0
#define RSBOI_WANT_PRINT_COMPLEX_OR_REAL 0
#define RSBOI_WANT_SUBSREF 1
#define RSBOI_WANT_HEAVY_DEBUG 0
#define RSBOI_WANT_VECLOAD_INSTEAD_MTX 1
//#define RSBOI_PERROR(E) rsb_perror(E)
#define RSBOI_PERROR(E) if(RSBOI_SOME_ERROR(E)) rsboi_strerr(E)
#ifdef RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX
#include <octave/ov-cx-mat.h>
#include <octave/ov-cx-sparse.h>
#endif

#ifndef RSBOI_RSB_MATRIX_SOLVE
#define RSBOI_RSB_MATRIX_SOLVE(V1,V2) RSBOI_0_ERROR(RSBOI_0_NOTERRMSG)  /* any solution routine shall attached here */
#endif

#if RSBOI_WANT_HEAVY_DEBUG
extern "C" {
	rsb_bool_t rsb_is_correctly_built_rcsr_matrix(const struct rsb_mtx_t *mtxAp); // forward declaration
}
#endif
#if defined(RSB_LIBRSB_VER) && (RSB_LIBRSB_VER>=10100)
extern "C" {
	int rsb_do_get_nnz_element(struct rsb_mtx_t*,void*,void*,void*,int);
}
#endif
#if RSBOI_WANT_DOUBLE_COMPLEX
#define RSBOI_BINOP_PREVAILING_TYPE(V1,V2) (((V1).is_complex_type()||(V2).is_complex_type())?RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX:RSB_NUMERICAL_TYPE_DOUBLE)
#else
#define RSBOI_BINOP_PREVAILING_TYPE(V1,V2) RSBOI_TYPECODE
#endif

void rsboi_strerr(rsb_err_t errval)
{
	const int errstrlen=128;
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

static const RSBOI_T rsboi_pone[]={+1.0,0.0}; 
static const RSBOI_T rsboi_mone[]={-1.0,0.0}; 
static const RSBOI_T rsboi_zero[]={ 0.0,0.0}; /* two elements, as shall work also with complex */

static octave_base_value * default_numeric_conversion_function (const octave_base_value& a);

static bool sparsersb_tester(void)
{
	if(sizeof(octave_idx_type)!=sizeof(rsb_coo_idx_t))
	{
		RSBOI_ERROR(RSBOI_0_INMISMMSG);
		goto err;
	}
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
	struct rsb_mtx_t * mtxAp;
	public:
		octave_sparsersb_mtx (void) : octave_sparse_matrix(RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			this->mtxAp=NULL;
		}

		octave_sparsersb_mtx (const octave_sparse_matrix &sm) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		}

		octave_sparsersb_mtx (const std::string &mtxfilename, rsb_type_t typecode=RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			rsb_err_t errval=RSB_ERR_NO_ERROR;
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

		//void alloc_rsb_mtx_from_coo_copy(const idx_vector &IM, const idx_vector &JM, const void * SMp, octave_idx_type nrA, octave_idx_type ncA, bool iscomplex=false, rsb_flags_t eflags=RSBOI_DCF)
		void alloc_rsb_mtx_from_coo_copy(idx_vector & IM, idx_vector & JM, const void * SMp, octave_idx_type nrA, octave_idx_type ncA, bool iscomplex=false, rsb_flags_t eflags=RSBOI_DCF)
		{
			octave_idx_type nnzA=IM.length();
			rsb_err_t errval=RSB_ERR_NO_ERROR;
#if RSBOI_WANT_DOUBLE_COMPLEX
			rsb_type_t typecode=iscomplex?RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX:RSB_NUMERICAL_TYPE_DOUBLE;
#else
			rsb_type_t typecode=RSBOI_TYPECODE;
#endif
			const rsb_coo_idx_t *IA=NULL,*JA=NULL;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
#if RSBOI_WANT_SYMMETRY
			/* shall verify if any symmetry is present */
#endif
			IA=(const rsb_coo_idx_t*)IM.raw();
		       	JA=(const rsb_coo_idx_t*)JM.raw();
			//RSB_DO_FLAG_ADD(eflags,rsb_util_determine_uplo_flags(IA,JA,nnzA));
			if(!(this->mtxAp = rsb_mtx_alloc_from_coo_const(SMp,IA,JA,nnzA,typecode,nrA,ncA,RSBOI_RB,RSBOI_CB,RSBOI_RF|eflags ,&errval)))
				RSBOI_ERROR(RSBOI_0_ALERRMSG);
			//RSBOI_MP(this->mtxAp);
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
			rsb_nnz_idx_t nnzA=0;
			Array<rsb_coo_idx_t> IA( dim_vector(1,sm.nnz()) );
			Array<rsb_coo_idx_t> JA( dim_vector(1,sm.nnz()) );
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			/* bool islowtri=sm.is_lower_triangular(),isupptri=sm.is_upper_triangular(); */
			rsb_flags_t eflags=RSBOI_RF;
			rsb_type_t typecode=RSB_NUMERICAL_TYPE_DOUBLE;
			octave_idx_type nrA = sm.rows (), ncA = sm.cols ();
#if RSBOI_WANT_SYMMETRY
			if(sm.is_symmetric())
				RSB_DO_FLAG_ADD(eflags,RSB_FLAG_LOWER_SYMMETRIC|RSB_FLAG_TRIANGULAR);
#endif
			if(!(this->mtxAp = rsb_mtx_alloc_from_csc_const(sm.data(),sm.ridx(),sm.cidx(), nnzA=sm.nnz(),typecode, nrA, ncA, RSBOI_RB, RSBOI_CB, eflags,&errval)))
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
			octave_idx_type nnzA=0;
			Array<rsb_coo_idx_t> IA( dim_vector(1,sm.nnz()) );
			Array<rsb_coo_idx_t> JA( dim_vector(1,sm.nnz()) );
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			/* bool islowtri=sm.is_lower_triangular(),isupptri=sm.is_upper_triangular(); */
			rsb_flags_t eflags=RSBOI_RF;
			rsb_type_t typecode=RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX;
#if RSBOI_WANT_SYMMETRY
			if(sm.is_hermitian())
				RSB_DO_FLAG_ADD(eflags,RSB_FLAG_LOWER_HERMITIAN|RSB_FLAG_TRIANGULAR);
#endif
			if(!(this->mtxAp = rsb_mtx_alloc_from_csc_const(sm.data(),sm.ridx(),sm.cidx(), nnzA=sm.nnz(),typecode, nrA, ncA, RSBOI_RB, RSBOI_CB, eflags,&errval)))
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

		octave_sparsersb_mtx (const SparseComplexMatrix &sm, rsb_type_t typecode=RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			this->alloc_rsb_mtx_from_csc_copy(sm);
		}
#endif

		octave_sparsersb_mtx (const SparseMatrix &sm, rsb_type_t typecode=RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			this->alloc_rsb_mtx_from_csc_copy(sm);
		}

		octave_sparsersb_mtx (struct rsb_mtx_t * mtxBp) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX), mtxAp(mtxBp)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			if(!this->mtxAp)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparsersb_mtx (const octave_sparsersb_mtx& T) :
		octave_sparse_matrix (T)  {
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			struct rsb_mtx_t*mtxBp=NULL;
		       	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			errval = rsb_mtx_clone(&mtxBp,RSB_NUMERICAL_TYPE_SAME_TYPE,RSB_TRANSPOSITION_N,NULL,T.mtxAp,RSBOI_EXPF);
			this->mtxAp=mtxBp;
		};
		octave_idx_type length (void) const { return this->nnz(); }
		octave_idx_type nelem (void) const { return this->nnz(); }
		octave_idx_type numel (void) const { return this->nnz(); }
		octave_idx_type nnz (void) const { rsb_nnz_idx_t nnzA=0; RSBOI_0_EMCHECK(this->mtxAp); rsb_mtx_get_info(this->mtxAp,RSB_MIF_MATRIX_NNZ__TO__RSB_NNZ_INDEX_T,&nnzA);  return nnzA;}
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

#if 0
		octave_value do_index_op(const octave_value_list& idx, bool resize_ok)
		{
			...
		}
#endif

		virtual SparseMatrix sparse_matrix_value(bool = false)const
		{
			struct rsboi_coo_matrix_t rcm;
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			rsb_nnz_idx_t nnzA,nzi;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			RSBOI_0_EMCHECK(this->mtxAp);
			nnzA=this->nnz();
			Array<octave_idx_type> IA( dim_vector(1,nnzA) );
			Array<octave_idx_type> JA( dim_vector(1,nnzA) );
			Array<RSBOI_T> VA( dim_vector(1,nnzA) );
			rcm.IA=(rsb_coo_idx_t*)IA.data(),rcm.JA=(rsb_coo_idx_t*)JA.data();
			if(!this->is_real_type())
			{
				Array<Complex> VAC( dim_vector(1,nnzA) );
				RSBOI_T* VAp=((RSBOI_T*)VA.data());
				rcm.VA=(RSBOI_T*)VAC.data();
#if RSBOI_WANT_SYMMETRY
				/* FIXME: and now ? shall we expand symmetry or not ? */
#endif
				/* FIXME: shall use some librsb's dedicated call for this */
				errval = rsb_mtx_get_coo(this->mtxAp,rcm.VA,rcm.IA,rcm.JA,RSB_FLAG_C_INDICES_INTERFACE);
				for(nzi=0;nzi<nnzA;++nzi)
					VAp[nzi]=((RSBOI_T*)rcm.VA)[2*nzi];
			}
			else
			{
				rcm.VA=(RSBOI_T*)VA.data();
				errval = rsb_mtx_get_coo(this->mtxAp,rcm.VA,rcm.IA,rcm.JA,RSB_FLAG_C_INDICES_INTERFACE);
			}
			rcm.nrA=this->rows();
			rcm.ncA=this->cols();
			return SparseMatrix(VA,IA,JA,rcm.nrA,rcm.ncA);
		}

		virtual Matrix matrix_value(bool = false)const
		{
			RSBOI_FIXME("inefficient!");
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			Matrix cm=this->sparse_matrix_value().matrix_value();
			return cm;
		}

		virtual octave_value full_value(void)const
		{
			RSBOI_FIXME("inefficient!");
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			if(this->is_real_type())
				return this->matrix_value();
			else
				return this->complex_matrix_value();
		}

#if RSBOI_WANT_DOUBLE_COMPLEX
		virtual ComplexMatrix complex_matrix_value(bool = false)const
		{
			RSBOI_FIXME("inefficient!");
			octave_sparse_complex_matrix ocm=this->sparse_complex_matrix_value();
			ComplexMatrix cm=ocm.complex_matrix_value();
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			return cm;
		}

		virtual SparseComplexMatrix sparse_complex_matrix_value(bool = false)const
		{
			struct rsboi_coo_matrix_t rcm;
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			rsb_nnz_idx_t nnzA,nzi;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			RSBOI_0_EMCHECK(this->mtxAp);
			nnzA=this->nnz();
			Array<octave_idx_type> IA( dim_vector(1,nnzA) );
			Array<octave_idx_type> JA( dim_vector(1,nnzA) );
			Array<Complex> VA( dim_vector(1,nnzA) );
			RSBOI_T* VAp=((RSBOI_T*)VA.data());
			rcm.IA=(rsb_coo_idx_t*)IA.data(),rcm.JA=(rsb_coo_idx_t*)JA.data();
			rcm.VA=VAp;
			errval = rsb_mtx_get_coo(this->mtxAp,rcm.VA,rcm.IA,rcm.JA,RSB_FLAG_C_INDICES_INTERFACE);
#if RSBOI_WANT_SYMMETRY
			/* FIXME: and now ? shall we expand symmetry or not ? */
#endif
			/* FIXME: shall use some librsb's dedicated call for this */
			if(this->is_real_type())
				for(nzi=0;nzi<nnzA;++nzi)
					VAp[2*(nnzA-1-nzi)+0]=VAp[(nnzA-1-nzi)+0],
					VAp[2*(nnzA-1-nzi)+1]=0;
			rcm.nrA=this->rows();
			rcm.ncA=this->cols();
			return SparseComplexMatrix(VA,IA,JA,rcm.nrA,rcm.ncA);
		}
#endif

		//octave_value::assign_op, int, int, octave_value (&)(const octave_base_value&, const octave_base_value&)
		//octave_value::assign_op, int, int, octave_value (&)
		//octave_value  assign_op (const octave_base_value&, const octave_base_value&) {}
		// octave_value::assign_op octave_value::binary_op_to_assign_op (binary_op op) { assign_op retval; return retval; }
#if RSBOI_WANT_SUBSREF
		octave_value subsref (const std::string &type, const std::list<octave_value_list>& idx)
		{
			octave_value retval;
			int skip = 1;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rsb_err_t errval=RSB_ERR_NO_ERROR;

			switch (type[0])
			{
				case '(':
				if (type.length () == 1)
				{
  					octave_idx_type n_idx = idx.front().length ();
					if (n_idx == 1 )
					{
						RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	    					idx_vector i = idx.front() (0).index_vector ();
#if   defined(RSB_LIBRSB_VER) && (RSB_LIBRSB_VER< 10100)
						octave_idx_type ii=i(0);
						RSBOI_ERROR("");
#elif defined(RSB_LIBRSB_VER) && (RSB_LIBRSB_VER>=10100)
						octave_idx_type ii=i(0);
						RSBOI_DEBUG_NOTICE("get_element (%d)\n",ii);
						if(is_real_type())
						{
							RSBOI_T rv;
							errval = rsb_do_get_nnz_element(this->mtxAp,&rv,NULL,NULL,ii);
							retval=rv;
						}
						else
						{
							Complex rv;
							errval = rsb_do_get_nnz_element(this->mtxAp,&rv,NULL,NULL,ii);
							retval=rv;
						}
						if(RSBOI_SOME_ERROR(errval))
						{
							if(ii>=this->nnz() || ii<0)
								error ("trying accessing element %d: index out of bounds !",ii+1);
							else
								error ("trying accessing element %d: this seems bug!",ii+1);
						}
#endif
					}
					else
					if (n_idx == 2 )
	  				{
	    					idx_vector i = idx.front() (0).index_vector ();
	    					if (! error_state)
	      					{
#if RSBOI_WANT_SYMMETRY
							/* FIXME: and now ? */
#endif
							if(is_real_type())
							{
								idx_vector j = idx.front() (1).index_vector ();
								RSBOI_T rv;
						  		octave_idx_type ii=-1,jj=-1;
  								ii=i(0); jj=j(0);
								RSBOI_DEBUG_NOTICE("get_elements (%d %d)\n",ii,jj);
       								errval = rsb_mtx_get_values(this->mtxAp,&rv,&ii,&jj,1,RSBOI_NF);
								retval=rv;
								if (! error_state)
								  ;
							}
							else
							{
								idx_vector j = idx.front() (1).index_vector ();
								Complex rv;
						  		octave_idx_type ii=-1,jj=-1;
  								ii=i(0); jj=j(0);
								RSBOI_DEBUG_NOTICE("get_elements (%d %d) complex\n",ii,jj);
       								errval = rsb_mtx_get_values(this->mtxAp,&rv,&ii,&jj,1,RSBOI_NF);
								retval=rv;
								if (! error_state)
								  ;
							}
	      					}
	  				}
				}
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
			if (! error_state)
				retval = retval.next_subsref (type, idx, skip);
			return retval;
		}
#else
		/* FIXME: need an alternative, bogus implementation of subsref */
#endif

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
		bool is_sparse_type (void) const { RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return true; }
		bool is_real_type (void) const { RSBOI_0_EMCHECK(this->mtxAp); RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return this->rsbtype()==RSB_NUMERICAL_TYPE_DOUBLE?true:false; }
		bool is_diagonal (void) const { RSBOI_0_EMCHECK(this->mtxAp); RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_DIAGONAL)?true:false; }/* FIXME: new: not sure whether this is ever called */
		bool is_lower_triangular (void) const { RSBOI_0_EMCHECK(this->mtxAp); RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_LOWER_TRIANGULAR)?true:false; }/* FIXME: new: not sure whether this is ever called */
		bool is_upper_triangular (void) const { RSBOI_0_EMCHECK(this->mtxAp); RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_UPPER_TRIANGULAR)?true:false; }/* FIXME: new: not sure whether this is ever called */
		bool is_complex_type (void) const { RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG); return !is_real_type(); }
		bool is_bool_type (void) const { return false; }
		bool is_integer_type (void) const { return false; }
		bool is_square (void) const { return this->rows()==this->cols(); }
		bool is_empty (void) const { return false; }
		/* bool is__symmetric (void) const { if(RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_SYMMETRIC))return true; return false; }*/ /* new */
		/* bool is__hermitian (void) const { if(RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_HERMITIAN))return true; return false; }*/ /* new */
		std::string get_symmetry (void) const { return (RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_SYMMETRIC)?"S": (RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_HERMITIAN)?"H":"U")); }
		bool is__triangular (void) const
	       	{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		       	if(this->mtxAp 
#if RSBOI_WANT_SYMMETRY
				&& ((!RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_SYMMETRIC)) || RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_DIAGONAL))
#endif
			  )
			{
				return RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_TRIANGULAR)?RSB_BOOL_TRUE:RSB_BOOL_FALSE;
			}
			else
			       	return RSB_BOOL_FALSE;
		}
//		int is_struct (void) const { return false; }

		octave_value subsasgn (const std::string& type, const std::list<octave_value_list>& idx, const octave_value& rhs)
		{
			octave_value retval;
			rsb_err_t errval=RSB_ERR_NO_ERROR;

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
							RSBOI_DEBUG_NOTICE("UNFINISHED\n");
							idx_vector i = idx.front()(0).index_vector ();
							if (! error_state)
								;//retval = octave_value (matrix.index (i, resize_ok));
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
									ComplexMatrix cm=rhs.complex_matrix_value();
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
								if (! error_state)
								{
									if(is_real_type())
									{
										rsb_err_t errval=RSB_ERR_NO_ERROR;
										octave_idx_type ii=-1,jj=-1;
										RSBOI_T rv=rhs.double_value();
										ii=i(0); jj=j(0);
										RSBOI_DEBUG_NOTICE("update elements (%d %d)\n",ii,jj);
#if RSBOI_WANT_SYMMETRY
										/* FIXME: and now ? */
#endif
										errval = rsb_mtx_set_values(this->mtxAp,&rv,&ii,&jj,1,RSBOI_NF);
										RSBOI_PERROR(errval);
										/* FIXME: I am unsure, here */
										//retval=rhs.double_value(); // this does not match octavej
										//retval=octave_value(this); 
										retval=octave_value(this->clone()); // matches but .. heavy ?!
										if (! error_state)
											;//retval = octave_value (matrix.index (i, j, resize_ok));
									}
									else
									{
										rsb_err_t errval=RSB_ERR_NO_ERROR;
										octave_idx_type ii=-1,jj=-1;
										Complex rv=rhs.complex_value();
										ii=i(0); jj=j(0);
										RSBOI_DEBUG_NOTICE("update elements (%d %d) complex\n",ii,jj);
#if RSBOI_WANT_SYMMETRY
				/* FIXME: and now ? */
#endif
										errval = rsb_mtx_set_values(this->mtxAp,&rv,&ii,&jj,1,RSBOI_NF);
										RSBOI_PERROR(errval);
										/* FIXME: I am unsure, here */
										//retval=rhs.double_value(); // this does not match octavej
										//retval=octave_value(this); 
										retval=octave_value(this->clone()); // matches but .. heavy ?!
										if (! error_state)
											;//retval = octave_value (matrix.index (i, j, resize_ok));
									}
//		  class Octave_map;
//		  retval = Octave_map();
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
		}

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

		void print (std::ostream& os, bool pr_as_read_syntax = false) const
		{
			RSBOI_FIXME("what to do with pr_as_read_syntax ?");
			struct rsboi_coo_matrix_t rcm;
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			rsb_nnz_idx_t nnzA=this->nnz(),nzi;
			bool ic=this->is_real_type()?false:true;
			Array<octave_idx_type> IA( dim_vector(1,nnzA) );
			Array<octave_idx_type> JA( dim_vector(1,nnzA) );
			Array<RSBOI_T> VA( dim_vector(1,(ic?2:1)*nnzA) );
			std::string c=ic?"complex":"real";
#if RSBOI_WANT_PRINT_DETAIL
			char ss[RSBOI_INFOBUF];
			rsb_mtx_get_info_str(this->mtxAp,"RSB_MIF_MATRIX_INFO__TO__CHAR_P",ss,RSBOI_INFOBUF);
#endif
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rcm.VA=(RSBOI_T*)VA.data(),rcm.IA=(rsb_coo_idx_t*)IA.data(),rcm.JA=(rsb_coo_idx_t*)JA.data();
#if RSBOI_WANT_SYMMETRY
			/* FIXME: and now ? */
#endif
			if(rcm.VA==NULL)
				nnzA=0;
			else
				errval = rsb_mtx_get_coo(this->mtxAp,rcm.VA,rcm.IA,rcm.JA,RSB_FLAG_C_INDICES_INTERFACE);
			rcm.nrA=this->rows();
			rcm.ncA=this->cols();
			double pct = 100.0*(((RSBOI_T)nnzA)/((RSBOI_T)rcm.nrA))/rcm.ncA;
			octave_stdout<<RSBOI_FSTR<< "  (rows = "<<rcm.nrA<<
				", cols = "<<rcm.ncA<<
				", nnz = "<<nnzA
#if RSBOI_WANT_SYMMETRY
				<< ", symm = "<<
				(RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_SYMMETRIC)?"S":
				(RSB_DO_FLAG_HAS(this->rsbflags(),RSB_FLAG_SYMMETRIC)?"H":"U"))
				<< // FIXME: need a mechanism to print out these flags from rsb itself
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
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			//RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			if(this->is_real_type())
			{
				Matrix DA(this->rows(),1);
				errval = rsb_mtx_get_vec(this->mtxAp,(RSBOI_T*)DA.data(),RSB_EXTF_DIAG);
				retval=(DA);
			}
			else
			{
				ComplexMatrix DA(this->rows(),1);
				errval = rsb_mtx_get_vec(this->mtxAp,(RSBOI_T*)DA.data(),RSB_EXTF_DIAG);
				retval=(DA);
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
		RSBOI_T one=1.0;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		return rsboi_get_scaled_copy(one/alpha);/* FIXME: is this correct ? */
	}

#if RSBOI_WANT_DOUBLE_COMPLEX
	octave_value rsboi_get_scaled_copy_inv(const Complex alpha)const
	{
		Complex one=1.0;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		return rsboi_get_scaled_copy(one/alpha);/* FIXME: is this correct ? */
	}
#endif

	octave_value rsboi_get_scaled_copy(const RSBOI_T alpha, rsb_trans_t transa=RSB_TRANSPOSITION_N)const
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		struct rsb_mtx_t*mtxBp=NULL;
		if(is_real_type())
		{
			errval = rsb_mtx_clone(&mtxBp,RSB_NUMERICAL_TYPE_SAME_TYPE,transa, &alpha,this->mtxAp,RSBOI_EXPF);
		}
		else
#if RSBOI_WANT_DOUBLE_COMPLEX
		{
			Complex calpha;calpha+=alpha;
			errval = rsb_mtx_clone(&mtxBp,RSB_NUMERICAL_TYPE_SAME_TYPE,transa,&calpha,this->mtxAp,RSBOI_EXPF);
		}
#else
		{RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);}
#endif
		return new octave_sparsersb_mtx( mtxBp );
	}

#if RSBOI_WANT_DOUBLE_COMPLEX
	octave_value rsboi_get_scaled_copy(const Complex alpha)const
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		octave_sparsersb_mtx * m = NULL;
		struct rsb_mtx_t*mtxBp=NULL;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		errval = rsb_mtx_clone(&mtxBp,RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX,RSB_TRANSPOSITION_N,&alpha,this->mtxAp,RSBOI_EXPF);
		m = new octave_sparsersb_mtx( mtxBp );
		return m;
	}
#endif

octave_value scale_rows(const octave_matrix&v2, bool want_div=false)
{
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	if(this->is_real_type())
	{
		const Matrix rm = want_div?1.0/v2.matrix_value ():v2.matrix_value ();
		octave_idx_type b_nc = rm.cols ();
		octave_idx_type b_nr = rm.rows ();
		octave_idx_type ldb=b_nr;
		octave_idx_type ldc=this->columns();
		octave_idx_type nrhs=b_nc;
		Matrix retval(ldc,nrhs,RSBOI_ZERO);
		if(this->rows()!=b_nr) { error("matrices dimensions do not match!\n"); return Matrix(); }
		errval=rsb_mtx_upd_values(this->mtxAp,RSB_ELOPF_SCALE_ROWS,rm.data());
		RSBOI_PERROR(errval);
		return retval;
	}
	else
	{
		const ComplexMatrix cm = want_div?1.0/v2.complex_matrix_value ():v2.complex_matrix_value ();
		octave_idx_type b_nc = cm.cols ();
		octave_idx_type b_nr = cm.rows ();
		octave_idx_type ldb=b_nr;
		octave_idx_type ldc=this->columns();
		octave_idx_type nrhs=b_nc;
		ComplexMatrix retval(ldc,nrhs,RSBOI_ZERO);
		if(this->rows()!=b_nr) { error("matrices dimensions do not match!\n"); return ComplexMatrix(); }
		errval=rsb_mtx_upd_values(this->mtxAp,RSB_ELOPF_SCALE_ROWS,cm.data());
		RSBOI_PERROR(errval);
		return retval;
	}
}

octave_value spmm(const octave_matrix&v2, bool do_trans=false)const
{
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	rsb_trans_t transa = do_trans ? RSB_TRANSPOSITION_T : RSB_TRANSPOSITION_N;

	if(this->is_real_type())
	{
		const Matrix b = v2.matrix_value ();
		octave_idx_type b_nc = b.cols ();
		octave_idx_type b_nr = b.rows ();
		octave_idx_type ldb=b_nr;
		octave_idx_type ldc=do_trans?this->columns():this->rows();
		octave_idx_type nrhs=b_nc;
		Matrix retval(ldc,nrhs,RSBOI_ZERO);
		if(this->columns()!=b_nr) { error("matrices dimensions do not match!\n"); return Matrix(); }
		if(( do_trans) &&(this->rows() !=b_nr)) { error("matrices dimensions do not match!\n"); return Matrix(); }
		if((!do_trans)&&(this->columns()!=b_nr)) { error("matrices dimensions do not match!\n"); return Matrix(); }
		errval=rsb_spmm(RSB_TRANSPOSITION_N,&rsboi_pone,this->mtxAp,nrhs,RSB_OI_DMTXORDER,(RSBOI_T*)b.data(),ldb,&rsboi_zero,(RSBOI_T*)retval.data(),ldc);
		RSBOI_PERROR(errval);
		return retval;
	}
	else
	{
		const ComplexMatrix b = v2.complex_matrix_value ();
		octave_idx_type b_nc = b.cols ();
		octave_idx_type b_nr = b.rows ();
		octave_idx_type ldb=b_nr;
		octave_idx_type ldc=do_trans?this->columns():this->rows();
		octave_idx_type nrhs=b_nc;
		ComplexMatrix retval(ldc,nrhs,RSBOI_ZERO);
		if(( do_trans) &&(this->rows() !=b_nr)) { error("matrices dimensions do not match!\n"); return ComplexMatrix(); }
		if((!do_trans)&&(this->columns()!=b_nr)) { error("matrices dimensions do not match!\n"); return ComplexMatrix(); }
		errval=rsb_spmm(RSB_TRANSPOSITION_N,&rsboi_pone,this->mtxAp,nrhs,RSB_OI_DMTXORDER,(RSBOI_T*)b.data(),ldb,&rsboi_zero,(RSBOI_T*)retval.data(),ldc);
		RSBOI_PERROR(errval);
		return retval;
	}
}

octave_value spmsp(const octave_sparsersb_mtx&v2)const
{
	rsb_err_t errval=RSB_ERR_NO_ERROR;
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

octave_value sppsp(const RSBOI_T*betap, const octave_sparsersb_mtx&v2)const
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	octave_sparsersb_mtx*sm = new octave_sparsersb_mtx();
	octave_value retval = sm;
	rsb_err_t errval=RSB_ERR_NO_ERROR;
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

octave_value cp_ubop(enum rsb_elopf_t opf, void*alphap=NULL)const
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	octave_sparsersb_mtx * m = new octave_sparsersb_mtx(*this);
	if(!m)return m;
	errval = rsb_mtx_upd_values(m->mtxAp,opf,alphap);
	RSBOI_PERROR(errval);
	return m;
}

	private:

		DECLARE_OCTAVE_ALLOCATOR
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

static octave_base_value * default_numeric_conversion_function (const octave_base_value& a)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_CONV_ARG (const octave_sparsersb_mtx&);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_WARN(RSBOI_0_UNFFEMSG);
	if(v.is_real_type())
		return new octave_sparse_matrix (v.sparse_matrix_value());
	else
		return new octave_sparse_complex_matrix (v.sparse_complex_matrix_value());
}

DEFINE_OCTAVE_ALLOCATOR (octave_sparsersb_mtx)
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA (octave_sparsersb_mtx,
RSB_OI_TYPEINFO_STRING,
RSB_OI_TYPEINFO_TYPE)

DEFCONV (octave_triangular_conv, octave_sparsersb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	CAST_CONV_ARG (const octave_sparsersb_mtx &);
	return new octave_sparse_matrix (v.matrix_value ());
}

#if 0
DEFCONV (octave_sparse_rsb_to_octave_sparse_conv, sparse_rsb_mtx, sparse_matrix)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_CONV_ARG (const octave_sparsersb_mtx &);
	return new octave_sparse_matrix (v.matrix_value ());
}
#endif

DEFUNOP (uplus, sparse_rsb_mtx)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	return new octave_sparsersb_mtx (v);
}

#if 0
DEFUNOP (op_incr, sparse_rsb_mtx)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	const octave_idx_type rn=v.mtxAp->nrA,cn=v.mtxAp->ncA;
	Matrix v2(rn,cn);
	octave_value retval=v2;
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	errval|=rsb_mtx_add_to_dense(&rsboi_pone,v.mtxAp,rn,rn,cn,RSB_BOOL_TRUE,(RSBOI_T*)v2.data());
	//v=octave_ma(idx, v2.matrix_value());
	return v2;
}

DEFUNOP (op_decr, sparse_rsb_mtx)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	const octave_idx_type rn=v.mtxAp->nrA,cn=v.mtxAp->ncA;
	Matrix v2(rn,cn);
	octave_value retval=v2;
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	errval|=rsb_mtx_add_to_dense(&rsboi_pone,v.mtxAp,rn,rn,cn,RSB_BOOL_TRUE,(RSBOI_T*)v2.data());
	//v=octave_ma(idx, v2.matrix_value());
	return v2;
}
#endif

DEFUNOP (uminus, sparse_rsb_mtx)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	return v.cp_ubop(RSB_ELOPF_NEG);
}

DEFUNOP (transpose, sparse_rsb_mtx)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	return v.rsboi_get_scaled_copy(rsboi_pone[0],RSB_TRANSPOSITION_T);
}

DEFUNOP (htranspose, sparse_rsb_mtx)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparsersb_mtx&);
	return v.rsboi_get_scaled_copy(rsboi_pone[0],RSB_TRANSPOSITION_C);
}

octave_value rsboi_spsv(const octave_sparsersb_mtx&v1, const octave_matrix&v2,rsb_trans_t transa)
{
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	if(v1.is_complex_type())
	{
	ComplexMatrix retval= v2.complex_matrix_value();
	octave_idx_type b_nc = retval.cols ();
	octave_idx_type b_nr = retval.rows ();
	octave_idx_type ldb=b_nr;
	octave_idx_type ldc=v1.rows();
	octave_idx_type nrhs=b_nc;
	octave_idx_type nels=retval.rows()*retval.cols();
	errval=rsb_spsm(transa,&rsboi_pone,v1.mtxAp,nrhs,RSB_OI_DMTXORDER,&rsboi_zero,(const RSBOI_T*)retval.data(),ldb,(RSBOI_T*)retval.data(),ldc);
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
	return retval;
	}
	else
	{
	Matrix retval=v2.matrix_value();
	octave_idx_type b_nc = retval.cols ();
	octave_idx_type b_nr = retval.rows ();
	octave_idx_type ldb=b_nr;
	octave_idx_type ldc=v1.rows();
	octave_idx_type nrhs=b_nc;
	octave_idx_type nels=retval.rows()*retval.cols();
	errval=rsb_spsm(transa,&rsboi_pone,v1.mtxAp,nrhs,RSB_OI_DMTXORDER,&rsboi_zero,(const RSBOI_T*)retval.data(),ldb,(RSBOI_T*)retval.data(),ldc);
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
	return retval;
	}
}

DEFBINOP(ldiv, sparse_rsb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_matrix&);
	if(v1.is__triangular()) 
		return rsboi_spsv(v1,v2,RSB_TRANSPOSITION_N);

	if(v1.is_complex_type() || v2.is_complex_type())
		return (v1.sparse_complex_matrix_value()).solve(v2.sparse_complex_matrix_value());
	else
		return (v1.sparse_matrix_value()).solve(v2.matrix_value());
	//RSBOI_RSB_MATRIX_SOLVE(v1,v2);
}

DEFBINOP(trans_ldiv, sparse_rsb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_matrix&);
	if(v1.is__triangular()) 
		return rsboi_spsv(v1,v2,RSB_TRANSPOSITION_T);

	if(v1.is_complex_type() || v2.is_complex_type())
		return (v1.sparse_complex_matrix_value().transpose()).solve(v2.sparse_complex_matrix_value());
	else
		return (v1.sparse_matrix_value().transpose()).solve(v2.matrix_value());
	//RSBOI_RSB_MATRIX_SOLVE(v1,v2);
}

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
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_complex&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.complex_value());
}
#endif

DEFBINOP(rsb_s_div, sparse_rsb_mtx, scalar)
{
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.scalar_value());
}

DEFBINOP(rsb_s_mul, sparse_rsb_mtx, scalar)
{
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy(v2.scalar_value());
}

#if RSBOI_WANT_DOUBLE_COMPLEX
DEFBINOP(rsb_c_mul, sparse_rsb_mtx, complex)
{
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_complex&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy(v2.complex_value());
}
#endif

#if 0
DEFBINOP(rsb_s_pow, sparse_rsb_mtx, scalar)
{
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_power_copy(v2.scalar_value());
}
#endif

DEFASSIGNOP (assign, sparse_rsb_mtx, sparse_rsb_mtx)
{
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	RSBOI_FIXME("I dunno how to trigger this!");
	CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_sparsersb_mtx&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	//rsb_assign(v1.mtxAp, v2.mtxAp);
	errval = rsb_mtx_clone(&v1.mtxAp,RSB_NUMERICAL_TYPE_SAME_TYPE,RSB_TRANSPOSITION_N,NULL,v2.mtxAp,RSBOI_EXPF);
	return octave_value();
}

DEFASSIGNOP (assignm, sparse_rsb_mtx, matrix)
{
	CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_matrix&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_DESTROY(v1.mtxAp);
	octave_value retval;
	//v1.assign(idx, v2.matrix_value());
	v1=(idx, v2.matrix_value());
	//retval=v1;
	retval=v2.matrix_value();
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return retval;
}

#if 0
DEFASSIGNOP(rsb_op_mul_eq_s, sparse_rsb_mtx, scalar)
{
	CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_scalar&);
	octave_value retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_PERROR(v1.rsboi_scale(v2.scalar_value()));
	retval=v1.matrix_value();
	return retval;
}

	rsb_err_t rsboi_scale(RSBOI_T alpha)
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		//errval=rsb_elemental_scale(this->mtxAp,&alpha);
	       	errval=rsb_elemental_op(this->mtxAp,RSB_ELOPF_MUL,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

	rsb_err_t rsboi_scale(Complex alpha)
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		//errval=rsb_elemental_scale(this->mtxAp,&alpha);
	       	errval=rsb_elemental_op(this->mtxAp,RSB_ELOPF_MUL,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

DEFASSIGNOP(rsb_op_div_eq_s, sparse_rsb_mtx, scalar)
{
	CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_scalar&);
	octave_value retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_PERROR(v1.rsboi_scale_inv(v2.scalar_value()));
	retval=v1.matrix_value();
	return retval;
}

	rsb_err_t rsboi_scale_inv(RSBOI_T alpha)
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		//errval=rsb_elemental_scale_inv(this->mtxAp,&alpha);
	       	errval=rsb_elemental_op(this->mtxAp,RSB_ELOPF_DIV,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

	rsb_err_t rsboi_scale_inv(Complex alpha)
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		//errval=rsb_elemental_scale_inv(this->mtxAp,&alpha);
	       	errval=rsb_elemental_op(this->mtxAp,RSB_ELOPF_DIV,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}
#endif

DEFBINOP(rsb_el_mul_s, sparse_rsb_mtx, scalar)
{
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy(v2.scalar_value());
}

#if RSBOI_WANT_DOUBLE_COMPLEX
DEFBINOP(rsb_el_mul_c, sparse_rsb_mtx, complex)
{
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_complex&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy(v2.complex_value());
}
#endif

DEFBINOP(rsb_el_div_s, sparse_rsb_mtx, scalar)
{
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.scalar_value());
}

#if RSBOI_WANT_DOUBLE_COMPLEX
DEFBINOP(rsb_el_div_c, sparse_rsb_mtx, complex)
{
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_complex&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.complex_value());
}
#endif

#if RSBOI_WANT_DOUBLE_COMPLEX
#if 0
DEFASSIGNOP(rsb_op_el_div_eq, sparse_rsb_mtx, scalar)
{
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	std::cout << "rsb_op_el_div_eq!\n";
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.complex_value());
}
#endif

DEFASSIGNOP(rsb_op_el_mul_eq_sc, sparse_rsb_mtx, matrix)
{
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_matrix&);
	return v1.scale_rows(v2,false);
}

DEFASSIGNOP(rsb_op_el_div_eq_sc, sparse_rsb_mtx, matrix)
{
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_matrix&);
	return v1.scale_rows(v2,true);
}
#endif

DEFBINOP(el_pow, sparse_rsb_mtx, scalar)
{
	CAST_BINOP_ARGS (const octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_T alpha=v2.scalar_value();
	return v1.cp_ubop(RSB_ELOPF_POW,&alpha);
}

#ifdef RSB_FULLY_IMPLEMENTED
DEFASSIGNOP (assigns, sparse_rsb_mtx, scalar)
{
	CAST_BINOP_ARGS (octave_sparsersb_mtx &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	v1.assign(idx, v2.matrix_value());
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return octave_value();
}
#endif

DEFBINOP(op_sub, sparse_rsb_mtx, sparse_rsb_mtx)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_sparsersb_mtx&);
	return v1.sppsp(&rsboi_mone[0],v2);
}

DEFBINOP(op_add, sparse_rsb_mtx, sparse_rsb_mtx)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_sparsersb_mtx&);
	return v1.sppsp(&rsboi_pone[0],v2);
}

DEFBINOP(op_spmul, sparse_rsb_mtx, sparse_rsb_mtx)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_sparsersb_mtx&);
	return v1.spmsp(v2);
}

DEFBINOP(op_mul, sparse_rsb_mtx, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_matrix&);
	return v1.spmm(v2, false);
}

DEFBINOP(op_trans_mul, sparse_rsb_mtx, matrix)
{
	// ".'*"  operator
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_BINOP_ARGS (const octave_sparsersb_mtx&, const octave_matrix&);
	return v1.spmm(v2, true);
}

static void install_sparsersb_ops (void)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	#ifdef RSB_FULLY_IMPLEMENTED
	/* boolean pattern-based not */
	INSTALL_UNOP (op_not, octave_sparsersb_mtx, op_not);
	/* to-dense operations */
	INSTALL_ASSIGNOP (op_asn_eq, octave_sparsersb_mtx, octave_scalar, assigns);
	/* ? */
	INSTALL_UNOP (op_uplus, octave_sparsersb_mtx, uplus);
	/* elemental comparison, evaluate to sparse or dense boolean matrices */
	INSTALL_BINOP (op_eq, octave_sparsersb_mtx, , );
	INSTALL_BINOP (op_le, octave_sparsersb_mtx, , );
	INSTALL_BINOP (op_lt, octave_sparsersb_mtx, , );
	INSTALL_BINOP (op_ge, octave_sparsersb_mtx, , );
	INSTALL_BINOP (op_gt, octave_sparsersb_mtx, , );
	INSTALL_BINOP (op_ne, octave_sparsersb_mtx, , );
	/* pure elemental; scalar and sparse arguments ?! */
								 // ?
	INSTALL_BINOP (op_el_ldiv, octave_sparsersb_mtx, , );
	INSTALL_BINOP (op_el_ldiv_eq, octave_sparsersb_mtx, , ); // errval=rsb_mtx_upd_values(this->mtxAp,RSB_ELOPF_SCALE_ROWS,cm.data());
	INSTALL_BINOP (op_el_mul_eq, octave_sparsersb_mtx, , ); // diagonal subst ??
	INSTALL_BINOP (op_el_and, octave_sparsersb_mtx, , );
	INSTALL_BINOP (op_el_or, octave_sparsersb_mtx, , );
	/* shift operations: they may be left out from the implementation */
	INSTALL_BINOP (op_lshift, octave_sparsersb_mtx, , );
	INSTALL_BINOP (op_rshift, octave_sparsersb_mtx, , );
	#endif
	// INSTALL_ASSIGNOP (op_el_div_eq, octave_sparsersb_mtx, octave_matrix, rsb_op_el_div_eq_sc); // errval=rsb_mtx_upd_values(this->mtxAp,RSB_ELOPF_SCALE_ROWS,cm.data());
	// INSTALL_ASSIGNOP (op_el_mul_eq, octave_sparsersb_mtx, octave_matrix, rsb_op_el_mul_eq_sc);
	//INSTALL_WIDENOP (octave_sparsersb_mtx, octave_sparse_matrix,octave_sparse_rsb_to_octave_sparse_conv);/* a DEFCONV .. */
	//INSTALL_ASSIGNCONV (octave_sparsersb_mtx, octave_sparse_matrix,octave_sparse_matrix);/* .. */
	// no need for the following: need a good conversion function, though
	//INSTALL_UNOP (op_incr, octave_sparsersb_mtx, op_incr);
	//INSTALL_UNOP (op_decr, octave_sparsersb_mtx, op_decr);
	INSTALL_BINOP (op_el_mul, octave_sparsersb_mtx, octave_scalar, rsb_el_mul_s);
#if RSBOI_WANT_DOUBLE_COMPLEX
	INSTALL_BINOP (op_el_mul, octave_sparsersb_mtx, octave_complex, rsb_el_mul_c);
#endif
//	INSTALL_ASSIGNOP (op_mul_eq, octave_sparsersb_mtx, octave_scalar, rsb_op_mul_eq_s); // 20110313 not effective
//	INSTALL_ASSIGNOP (op_div_eq, octave_sparsersb_mtx, octave_scalar, rsb_op_div_eq_s); // 20110313 not effective
	INSTALL_BINOP (op_el_div, octave_sparsersb_mtx, octave_scalar, rsb_el_div_s);
#if RSBOI_WANT_DOUBLE_COMPLEX
	INSTALL_BINOP (op_el_div, octave_sparsersb_mtx, octave_complex, rsb_el_div_c);
#endif
	INSTALL_BINOP (op_el_pow, octave_sparsersb_mtx, octave_scalar, el_pow);
	INSTALL_UNOP (op_uminus, octave_sparsersb_mtx, uminus);
	INSTALL_BINOP (op_ldiv, octave_sparsersb_mtx, octave_matrix, ldiv);
	INSTALL_BINOP (op_el_ldiv, octave_sparsersb_mtx, octave_matrix, el_ldiv);
	INSTALL_BINOP (op_div, octave_sparsersb_mtx, octave_matrix, div);
	INSTALL_BINOP (op_div, octave_sparsersb_mtx, octave_scalar, rsb_s_div);
#if RSBOI_WANT_DOUBLE_COMPLEX
	INSTALL_BINOP (op_div, octave_sparsersb_mtx, octave_complex, rsb_c_div);
#endif
	INSTALL_BINOP (op_mul, octave_sparsersb_mtx, octave_scalar, rsb_s_mul);
#if RSBOI_WANT_DOUBLE_COMPLEX
	INSTALL_BINOP (op_mul, octave_sparsersb_mtx, octave_complex, rsb_c_mul);
#endif
	//INSTALL_BINOP (op_pow, octave_sparsersb_mtx, octave_scalar, rsb_s_pow);
	INSTALL_BINOP (op_el_div, octave_sparsersb_mtx, octave_matrix, el_div);
	INSTALL_UNOP (op_transpose, octave_sparsersb_mtx, transpose);
	INSTALL_UNOP (op_hermitian, octave_sparsersb_mtx, htranspose);
	INSTALL_ASSIGNOP (op_asn_eq, octave_sparsersb_mtx, octave_sparse_matrix, assign);
	INSTALL_ASSIGNOP (op_asn_eq, octave_sparsersb_mtx, octave_matrix, assignm);
	INSTALL_BINOP (op_mul, octave_sparsersb_mtx, octave_matrix, op_mul);
	//INSTALL_BINOP (op_pow, octave_sparsersb_mtx, octave_matrix, op_pow);
	INSTALL_BINOP (op_sub, octave_sparsersb_mtx, octave_sparsersb_mtx, op_sub);
	INSTALL_BINOP (op_add, octave_sparsersb_mtx, octave_sparsersb_mtx, op_add);
	//INSTALL_BINOP (op_trans_add, octave_sparsersb_mtx, octave_sparsersb_mtx, op_trans_add);
	INSTALL_BINOP (op_mul, octave_sparsersb_mtx, octave_sparsersb_mtx, op_spmul);
	INSTALL_BINOP (op_trans_mul, octave_sparsersb_mtx, octave_matrix, op_trans_mul);
	INSTALL_BINOP (op_trans_ldiv, octave_sparsersb_mtx, octave_matrix, trans_ldiv);
	//INSTALL_BINOP (op_mul_trans, octave_sparsersb_mtx, octave_matrix, op_mul_trans);
	//INSTALL_BINOP (op_mul_trans, octave_sparsersb_mtx, octave_matrix, op_mul_trans);
	//INSTALL_BINOP (op_herm_mul, octave_sparsersb_mtx, octave_matrix, op_herm_mul);
	//INSTALL_BINOP (op_mul_herm, octave_sparsersb_mtx, octave_matrix, op_mul_herm);
	//INSTALL_BINOP (op_el_not_and, octave_sparsersb_mtx, octave_matrix, op_el_not_and);
	//INSTALL_BINOP (op_el_not_or , octave_sparsersb_mtx, octave_matrix, op_el_not_or );
	//INSTALL_BINOP (op_el_and_not, octave_sparsersb_mtx, octave_matrix, op_el_and_not);
	//INSTALL_BINOP (op_el_or _not, octave_sparsersb_mtx, octave_matrix, op_el_or _not);
}

static void install_sparse_rsb (void)
{
	static bool rsboi_initialized=false;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	if(!rsboi_initialized)
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		if(sparsersb_tester()==false)
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
		rsboi_initialized=true;
	}
	else
		;/* already initialized */

	if (!rsboi_sparse_rsb_loaded)
	{
		octave_sparsersb_mtx::register_type ();
		install_sparsersb_ops ();
		rsboi_sparse_rsb_loaded=true;
		mlock();
	}
	return;
	err:
	RSBIO_NULL_STATEMENT_FOR_COMPILER_HAPPINESS
}

DEFUN_DLD (RSB_SPARSERSB_LABEL, args, nargout,
"-*- texinfo -*-\n\
@deftypefn {Loadable Function} {@var{s} =} "RSBOI_FNS" (@var{a})\n\
Create a sparse RSB matrix from the full matrix @var{a}.\n"\
/*is forced back to a full matrix if resulting matrix is sparse\n*/\
"\n\
@deftypefnx {Loadable Function} {[@var{s}, @var{nrows}, @var{ncols}, @var{nnz}, @var{repinfo}, @var{field}, @var{symmetry}] =} "RSBOI_FNS" (@var{mtxfilename}, @var{mtxtypestring})\n\
Create a sparse RSB matrix by loading the Matrix Market matrix file named @var{mtxfilename}. The optional argument {@var{mtxtypestring}} can specify either real (\"D\") or complex (\"Z\") type. Default is real.\n"\
"In the case @var{mtxfilename} is \""RSBOI_LIS"\", a string listing the available numerical types with BLAS-style characters will be returned. If the file turns out to contain a Matrix Market vector, this will be loaded as such.\n"\
"\n\
@deftypefnx {Loadable Function} {@var{s} =} "RSBOI_FNS" (@var{i}, @var{j}, @var{sv}, @var{m}, @var{n}, @var{nzmax})\n\
Create a sparse RSB matrix given integer index vectors @var{i} and @var{j},\n\
a 1-by-@code{nnz} vector of real of complex values @var{sv}, overall\n\
dimensions @var{m} and @var{n} of the sparse matrix.  The argument\n\
@code{nzmax} is ignored but accepted for compatibility with @sc{Matlab}.\n\
\n\
@strong{Note}: if multiple values are specified with the same\n\
@var{i}, @var{j} indices, the corresponding values in @var{s} will\n\
be added.\n\
\n\
The following are all equivalent:\n\
\n\
@example\n\
@group\n\
s = "RSBOI_FNS" (i, j, s, m, n)\n\
s = "RSBOI_FNS" (i, j, s, m, n, \"summation\")\n\
s = "RSBOI_FNS" (i, j, s, m, n, \"sum\")\n"\
/*"s = "RSBOI_FNS" (i, j, s, \"summation\")\n"*/\
/*"s = "RSBOI_FNS" (i, j, s, \"sum\")\n"*/\
"@end group\n\
@end example\n\
\n\
@deftypefnx {Loadable Function} {@var{s} =} "RSBOI_FNS" (@var{i}, @var{j}, @var{s}, @var{m}, @var{n}, \"unique\")\n\
Same as above, except that if more than two values are specified for the\n\
same @var{i}, @var{j} indices, the last specified value will be used.\n\
\n\
@deftypefnx {Loadable Function} {@var{s} =} "RSBOI_FNS" (@var{i}, @var{j}, @var{sv})\n\
Uses @code{@var{m} = max (@var{i})}, @code{@var{n} = max (@var{j})}\n\
\n\
@deftypefnx {Loadable Function} {@var{s} =} "RSBOI_FNS" (@var{m}, @var{n})\n\
If @var{m} and @var{n} are integers, equivalent to @code{"RSBOI_FNS" ([], [], [], @var{m}, @var{n}, 0)}\n\
\n\
@deftypefnx {Loadable Function} {@var{s} =} "RSBOI_FNS" (\"set\", @var{opn}, @var{opv})\n\
If @var{opn} is a string representing a valid librsb option name and @var{opv} is a string representing a valid librsb option value, the correspondent librsb option will be set to that value.\n\
\n\
@deftypefnx {Loadable Function} {@var{s} =} "RSBOI_FNS" (@var{A}, \"get\", @var{mif})\n\
If @var{mif} is a string specifying a valid librsb matrix info string (valid for librsb's rsb_mtx_get_info_from_string()), then the correspondent value will be returned for matrix @var{A}. If @var{mif} is the an empty string (\"\"), matrix structure information will be returned.\n\
\n\
@deftypefnx {Loadable Function} {@var{s} =} "RSBOI_FNS" (@var{A}, @var{S})\n\
If @var{A} is a "RSBOI_FNS" matrix and @var{S} is a string, @var{S} will be interpreted as a query string about matrix @var{A}.\n\
\n"\
/*If any of @var{sv}, @var{i} or @var{j} are scalars, they are expanded\n\ 
to have a common size.\n*/
"\n\
Please note that on @code{"RSBOI_FNS"} type variables are available most, but not all of the operators available for @code{full} or @code{sparse} typed variables.\n\
\n\
@seealso{full, sparse}\n\
@end deftypefn")
{
	int nargin = args.length ();
	octave_value_list retval;
	octave_sparsersb_mtx*osmp=NULL;
	bool ic0=nargin>0?(args(0).is_complex_type()):false;
	bool ic3=nargin>2?(args(2).is_complex_type()):false;
	bool isr=(nargin>0 && args(0).type_name()==RSB_OI_TYPEINFO_STRING);

	RSBOI_DEBUG_NOTICE("in sparsersb()\n");

	//if(ic3 || ic0)
	if(ic0)
	{
		RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	}

	if(isr)
		osmp=((octave_sparsersb_mtx*)(args(0).internal_rep()));

	if(ic3 || ic0)
#if RSBOI_WANT_DOUBLE_COMPLEX
		RSBOI_WARN(RSBOI_0_UNCFEMSG);
#else
		RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);
#endif
	install_sparse_rsb();
	if( nargin == 3 && args(0).is_string() && args(0).string_value()=="set" && args(1).is_string() && args(2).is_string())
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		const char *os=args(1).string_value().c_str();
		const char *ov=args(2).string_value().c_str();
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
		error("did you intend to set librsb options ? use the correct syntax then !"); goto errp;
	}

	if( nargin == 2 && args(0).is_string() && args(0).string_value()=="get" && args(1).is_string() )
	{
		/* FIXME: unfinished feature ! */
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		error("getting library options still unimplemented!");
		goto errp;
	}

	if (nargin == 3 && isr 
 		&& args(1).is_string() && args(1).string_value()=="get"
		&& args(2).is_string())
	{
		/* FIXME: undocumented feature */
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		const char *mis=args(2).string_value().c_str();
		/* rsb_real_t miv=RSBOI_ZERO;*/ /* FIXME: this is extreme danger! */
		char ss[RSBOI_INFOBUF];
		if(!osmp || !osmp->mtxAp)
			goto ret;/* FIXME: error handling missing here */
		if(strlen(mis)==0)
		{
			mis="RSB_MIF_MATRIX_INFO__TO__CHAR_P";
		}
		errval = rsb_mtx_get_info_str(osmp->mtxAp,mis,ss,RSBOI_INFOBUF);

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
		error("did you intend to get matrices information ? use the correct syntax then !"); goto errp;
	}

	if ( nargin == 1 || nargin == 2 )
	{
		rsb_type_t typecode=RSBOI_TYPECODE;
		if (nargin >= 2)/* FIXME: this is weird ! */
#if RSBOI_WANT_DOUBLE_COMPLEX
			typecode=RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX;
#else
			RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);
#endif

		if (nargin == 2 && isr && args(1).is_string())
		{
			char ss[RSBOI_INFOBUF];
			if(!osmp || !osmp->mtxAp)goto ret;/* FIXME: error handling missing here */
			rsb_mtx_get_info_str(osmp->mtxAp,"RSB_MIF_MATRIX_INFO__TO__CHAR_P",ss,RSBOI_INFOBUF);
			/* FIXME: to add interpretation */
			RSBOI_WARN(RSBOI_0_UNFFEMSG);/* FIXME: this is yet unfinished */
			octave_stdout << "Matrix information (in the future, supplementary information may be returned, as more inquiry functionality will be implemented):\n" << ss << "\n";
			/* FIXME: shall not print out, but rather return the info as a string*/
			//retval.append("place info string here !\n");
			goto ret;
		}
		else
		if(args(0).is_sparse_type())
		{
			if( isr )
			{
				RSBOI_WARN(RSBOI_0_UNFFEMSG);
				retval.append(osmp=(octave_sparsersb_mtx*)(args(0).get_rep()).clone());
			}
			else
			{
				if(!ic0)
				{
					const SparseMatrix m = args(0).sparse_matrix_value();
					if (error_state) goto err;
					retval.append(osmp=new octave_sparsersb_mtx(m,typecode));
				}
#if RSBOI_WANT_DOUBLE_COMPLEX
				else
				{
					const SparseComplexMatrix m = args(0).sparse_complex_matrix_value();
					if (error_state) goto err;
					retval.append(osmp=new octave_sparsersb_mtx(m,typecode));
				}
#endif
			}
		}
		else
		if(args(0).is_string())
		{
			const std::string mtxfilename = args(0).string_value();
			if (error_state) goto err;
			if(mtxfilename==RSBOI_LIS)
			{
				//retval.append(RSB_NUMERICAL_TYPE_PREPROCESSOR_SYMBOLS);
#if RSBOI_WANT_DOUBLE_COMPLEX
				retval.append("D Z");
#else
				retval.append("D");
#endif
				goto ret;
			}
			else
			{
				rsb_type_t typecode=RSBOI_TYPECODE;
				RSBOI_WARN(RSBOI_0_UNFFEMSG);
				RSBOI_WARN("shall set the type, here");
				if(nargin>1 && args(1).is_string())
				{
					const std::string mtxtypestring = args(1).string_value();
					if(mtxtypestring=="complex" || mtxtypestring=="Z")
#if RSBOI_WANT_DOUBLE_COMPLEX
						typecode=RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX;

#else
						RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);
#endif
					if(mtxtypestring=="real" || mtxtypestring=="D")
						typecode=RSB_NUMERICAL_TYPE_DOUBLE;
				}
				osmp=new octave_sparsersb_mtx(mtxfilename,typecode);
				if(osmp->mtxAp)
					retval.append(osmp);
				else
					delete osmp;
#if RSBOI_WANT_VECLOAD_INSTEAD_MTX
				if(!osmp->mtxAp)
                		{
					rsb_nnz_idx_t n=0;
					rsb_file_vec_load(mtxfilename.c_str(),typecode,NULL,&n);
					if(n<1)
					{
						/* FIXME: message needed here */
						goto err;
					}

					if(typecode==RSB_NUMERICAL_TYPE_DOUBLE)
					{
						Matrix retvec(n,1,RSBOI_ZERO);
						rsb_file_vec_load(mtxfilename.c_str(),typecode,(RSBOI_T*)retvec.data(),&n);
						retval.append(retvec);
					}
#if RSBOI_WANT_DOUBLE_COMPLEX
					else
					if(typecode==RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX)
					{
						ComplexMatrix retvec(n,1,RSBOI_ZERO);
						rsb_file_vec_load(mtxfilename.c_str(),typecode,(RSBOI_T*)retvec.data(),&n);
						retval.append(retvec);
					}
#endif
					goto ret;
				}
#endif
				if(nargout) nargout--;
				if(nargout) retval.append(osmp->rows()),--nargout;
				if(nargout) retval.append(osmp->cols()),--nargout;
				if(nargout) retval.append(osmp->nnz()),--nargout;
				if(nargout) retval.append(osmp->get_info_string()),--nargout;
				if(nargout) retval.append((!osmp->is_complex_type())?"real":"complex"),--nargout;
				if(nargout) retval.append(osmp->get_symmetry()),--nargout;
			}
		}
		else
		{
			if(!ic0)
			{
				Matrix m = args(0).matrix_value();
				if (error_state) goto err;
				retval.append(osmp=new octave_sparsersb_mtx(m));
			}
#if RSBOI_WANT_DOUBLE_COMPLEX
			else
			{
				ComplexMatrix m = args(0).complex_matrix_value();
				if (error_state) goto err;
				retval.append(osmp=new octave_sparsersb_mtx(m));
			}
#endif
		}
	}
	else
	if (nargin >= 3 && nargin <= 6 && !(args(0).is_string() || args(1).is_string() || args(2).is_string() ) )
	{
		rsb_flags_t eflags=RSBOI_DCF;
		octave_idx_type nrA=0,ncA=0;
		if (nargin > 3)
		{
			if ( nargin < 5)
			{
//				if(nargin==4 && args(3).is_string())
//					goto checked;
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
//checked:
		if (nargin >= 5  )
		{
			nrA=args(3).scalar_value();/* FIXME: need index value here! */
			ncA=args(4).scalar_value();
			if(nrA<=0 || ncA<=0)
			{
				RSBOI_EERROR(RSBOI_O_NPMSERR);
				goto errp;
			}
		}
		if (nargin >= 6  && args(5).is_string())
		{
			std::string vv= args(5).string_value();
			if ( vv == "summation" || vv == "sum" )
				eflags=RSB_FLAG_DUPLICATES_SUM;
			else
			if ( vv == "unique" )
				eflags=RSB_FLAG_DUPLICATES_KEEP_LAST;
			else
				goto errp;
		}
		if (nargin >= 6  && args(5).is_integer_type())
		{
			/* we ignore this value for MATLAB compatibility */
		}
		if (error_state) goto ret;

		if(!ic3)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			idx_vector iv=args(0).index_vector ();
			idx_vector jv=args(1).index_vector ();
			retval.append(osmp=new octave_sparsersb_mtx( iv, jv, args(2).matrix_value(),nrA,ncA,eflags ));
		}

#if RSBOI_WANT_DOUBLE_COMPLEX
		else
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			idx_vector iv=args(0).index_vector ();
			idx_vector jv=args(1).index_vector ();
			retval.append(osmp=new octave_sparsersb_mtx( iv, jv, args(2).complex_matrix_value(),nrA,ncA,eflags ));
		}
#endif
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
