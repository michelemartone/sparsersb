/*
 Copyright (C) 2011   Michele Martone   <michele.martone@ipp.mpg.de>

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
 * TODO:
 * behaviour with complex data is often undefined. shall fix this: need complex support.
 * should implement compound_binary_op
 * for future versions, see http://www.gnu.org/software/octave/doc/interpreter/index.html#Top
 * for thorough testing, see Octave's test/build_sparse_tests.sh
 * need introspection functionality (bytes/nnz, or  sparsersb(rsbmat,"inquire: subm") )
 * sparsersb(rsbmat,"benchmark")
 * sparsersb(rsbmat,"test")
 * need properly working all scaling operations for complex
 * shall merge index/format/type conversions into librsb functionality:
 *  - conversion with indices adjustments and triangle computation:
 *    - from csc 
 *    - from coo 
 *    - from double to complex and viceversa, when calling rsb_get_coo
 *  - minimize copies around
 * subsref, dotref, subsasgn are incomplete: need error messages there
 * in full_value(), bool arg is ignored
 * missing symmetry support (although librsb has it)!
 * shall document the semantics of the update and access operators
 * r=0;r=sparsersb([1+1i]),r*=(2+i) changes the format of r
 * shall create a single standard error macro for constructors
 * shall test sistematically all constructors
 * often missing array lenghts/type checks
 * may define as map (see is_map) so that "a.type = ..." can work
 * is_struct, find_nonzero_elem_idx  are undefined
 * are octave_triangular_conv, default_numeric_conversion_function ok ? 
 *
 * Developer notes:
 /usr/share/doc/octave3.2-htmldoc//interpreter/Getting-Started-with-Oct_002dFiles.html#Getting-Started-with-Oct_002dFiles
 http://octave.sourceforge.net/developers.html
 * */

#include <octave/oct.h>
#include <octave/ov-re-mat.h>
#include <octave/ov-re-sparse.h>
#include <octave/ov-scalar.h>
#include <octave/ops.h>
#include <octave/ov-typeinfo.h>
#include <rsb.h>

#ifdef RSBOI_VERBOSE_CONFIG
#if (RSBOI_VERBOSE_CONFIG>0)
#define RSBOI_VERBOSE RSBOI_VERBOSE_CONFIG
#endif
#endif

#if 0
#define RSBOI_WARN( MSG ) \
	octave_stdout << "Warning in "<<__func__<<"(), in file "<<__FILE__<<" at line "<<__LINE__<<":\n" << MSG;
#else
#define RSBOI_WARN( MSG )
#endif
#define RSBOI_TODO( MSG ) RSBOI_WARN( MSG )/* new */

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
#define RSBOI_T double
#define RSBOI_MP(M) RSBOI_DUMP(RSB_PRINTF_MATRIX_SUMMARY_ARGS(M))
#undef RSB_FULLY_IMPLEMENTED
#define RSBOI_DESTROY(OM) {rsb_free_sparse_matrix(OM);(OM)=NULL;}
#define RSBOI_SOME_ERROR(ERRVAL) (ERRVAL)!=RSB_ERR_NO_ERROR
#define RSBOI_0_ERROR error
#define RSBOI_0_BADINVOERRMSG "invoking this function in the wrong way!\n"
#define RSBOI_0_ALLERRMSG "error allocating matrix!\n"
#define RSBOI_0_NOCOERRMSG "compiled without complex type support!\n"
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
#define RSBOI_D_EMPTY_MSG  "complex support is yet incomplete\n"
#define RSBOI_O_MISSIMPERRMSG  "implementation missing here\n"
#define RSBOI_O_NPMSERR  "providing non positive matrix size is not allowed!"
#define RSBOI_0_EMCHECK(M) if(!(M))RSBOI_0_ERROR(RSBOI_0_EMERRMSG);
#define RSBOI_FNSS(S)	#S
//#define RSBOI_FNS	RSBOI_FNSS(RSB_SPARSERSB_LABEL)
#define RSBOI_FSTR	"Recursive Sparse Blocks"
#define RSBOI_FNS	"sparsersb"
#define RSBOI_LIS	"?"

#define RSBIO_DEFAULT_CORE_MATRIX  Matrix (0,0)
/* FIXME : octave_idx_type vs rsb_coo_index_t */
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

#define RSBOI_WANT_SUBSREF 1
#define RSBOI_WANT_HEAVY_DEBUG 0
//#define RSBOI_PERROR(E) rsb_perror(E)
#define RSBOI_PERROR(E) if(RSBOI_SOME_ERROR(E))octave_stdout<<"librsb error:"<<rsb_strerror(E)<<"\n"
#define RSBOI_WANT_IDX_VECTOR_CONST 1


#ifdef RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX
#include <octave/ov-cx-mat.h>
#include <octave/ov-cx-sparse.h>
#endif

#if RSBOI_WANT_HEAVY_DEBUG
extern "C" {
	rsb_bool_t rsb_is_correctly_built_rcsr_matrix(const struct rsb_matrix_t *matrix); // forward declaration
}
#endif

struct rsb_coo_matrix_t
{
	octave_idx_type * IA, * JA;	 /** row and columns indices */
	octave_idx_type m,k;		 /** matrix (declared) nonzeros */
	octave_idx_type nnz;		 /** matrix rows, columns */
	void * VA;					 /** values of data elements */
	rsb_type_t typecode;		 /** as specified in the RSB_NUMERICAL_TYPE_* preprocessor symbols in types.h 	*/
};

static const RSBOI_T rsboi_one = 1.0; 
static const RSBOI_T rsboi_mone=-1.0; 
static const RSBOI_T rsboi_zero= 0.0; 

static octave_base_value * default_numeric_conversion_function (const octave_base_value& a);

static bool sparsersb_tester(void)
{
	if(sizeof(octave_idx_type)!=sizeof(rsb_coo_index_t))
	{
		RSBOI_ERROR(RSBOI_0_INMISMMSG);
	//	RSBOI_O_ERROR(RSBOI_0_INMISMMSG);
		goto err;
	}
	RSBOI_WARN(RSBOI_0_INCFERRMSG);
	return true;
err:
	return false;
}

static bool rsboi_sparse_rsb_loaded = false;

class octave_sparse_rsb_matrix : public octave_sparse_matrix
{
	private:
	public:
	struct rsb_matrix_t * A;
	public:
		octave_sparse_rsb_matrix (void) : octave_sparse_matrix(RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			this->A=NULL;
		}

		octave_sparse_rsb_matrix (const octave_sparse_matrix &sm) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		}

		octave_sparse_rsb_matrix (const std::string &fn, rsb_type_t typecode=RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			if(!(this->A=rsb_load_matrix_file_as_matrix_market(fn.c_str(),RSBOI_RF,typecode,&errval)))
				RSBOI_ERROR(RSBOI_0_ALERRMSG);
			RSBOI_PERROR(errval);
			if(!this->A)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		//void rsboi_allocate_rsb_matrix_from_coo_copy(const idx_vector &IM, const idx_vector &JM, const void * SMp, octave_idx_type nr, octave_idx_type nc, bool iscomplex=false, rsb_flags_t eflags=RSBOI_DCF)
#if RSBOI_WANT_IDX_VECTOR_CONST
		void rsboi_allocate_rsb_matrix_from_coo_copy(idx_vector & IM, idx_vector & JM, const void * SMp, octave_idx_type nr, octave_idx_type nc, bool iscomplex=false, rsb_flags_t eflags=RSBOI_DCF)
#else
		void rsboi_allocate_rsb_matrix_from_coo_copy(const Matrix &IM, const Matrix &JM, const void * SMp, octave_idx_type nr, octave_idx_type nc, bool iscomplex=false, rsb_flags_t eflags=RSBOI_DCF)
#endif
		{
#if RSBOI_WANT_IDX_VECTOR_CONST
			octave_idx_type nnz=IM.length();
#else
			octave_idx_type nnz=IM.rows()*IM.cols();
			Array<rsb_coo_index_t> IAv( dim_vector(1,nnz) );
			Array<rsb_coo_index_t> JAv( dim_vector(1,nnz) );
#endif
			rsb_err_t errval=RSB_ERR_NO_ERROR;
#if RSBOI_WANT_DOUBLE_COMPLEX
			rsb_type_t typecode=iscomplex?RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX:RSB_NUMERICAL_TYPE_DOUBLE;
#else
			rsb_type_t typecode=RSBOI_TYPECODE;
#endif
			const rsb_coo_index_t *IA=NULL,*JA=NULL;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

#if RSBOI_WANT_IDX_VECTOR_CONST
			IA=(const rsb_coo_index_t*)IM.raw();
		       	JA=(const rsb_coo_index_t*)JM.raw();
#else
			IA=(const rsb_coo_index_t*)IM.data();
			JA=(const rsb_coo_index_t*)JM.data();
#endif
			RSB_DO_FLAG_ADD(eflags,rsb_util_determine_uplo_flags(IA,JA,nnz));
			if(!(this->A=rsb_allocate_rsb_sparse_matrix_const(SMp,IA,JA,nnz,typecode,nr,nc,RSBOI_RB,RSBOI_CB,RSBOI_RF|eflags ,&errval)))
				RSBOI_ERROR(RSBOI_0_ALERRMSG);
			RSBOI_MP(this->A);
			RSBOI_MP(this->A);
			RSBOI_PERROR(errval);
			if(!this->A)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

#if RSBOI_WANT_DOUBLE_COMPLEX
#if RSBOI_WANT_IDX_VECTOR_CONST
		octave_sparse_rsb_matrix (idx_vector &IM, idx_vector &JM, const ComplexMatrix &SM,
			octave_idx_type nr, octave_idx_type nc, rsb_flags_t eflags) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rsboi_allocate_rsb_matrix_from_coo_copy(IM,JM,SM.data(),nr,nc,true,eflags);
		}
#else
		octave_sparse_rsb_matrix (const Matrix &IM, const Matrix &JM, const ComplexMatrix &SM,
			octave_idx_type nr, octave_idx_type nc, rsb_flags_t eflags) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rsboi_allocate_rsb_matrix_from_coo_copy(IM,JM,SM.data(),nr,nc,true,eflags);
		}
#endif
#endif

#if RSBOI_WANT_IDX_VECTOR_CONST
		octave_sparse_rsb_matrix (idx_vector &IM, idx_vector &JM, const Matrix &SM,
			octave_idx_type nr, octave_idx_type nc, rsb_flags_t eflags) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rsboi_allocate_rsb_matrix_from_coo_copy(IM,JM,SM.data(),nr,nc,false,eflags);
		}
#else
		octave_sparse_rsb_matrix (const Matrix &IM, const Matrix &JM, const Matrix &SM,
			octave_idx_type nr, octave_idx_type nc, rsb_flags_t eflags) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rsboi_allocate_rsb_matrix_from_coo_copy(IM,JM,SM.data(),nr,nc,false,eflags);
		}
#endif

		void rsboi_allocate_rsb_matrix_from_csc_copy(const SparseMatrix &sm)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			octave_idx_type nnz=0;
			Array<rsb_coo_index_t> IA( dim_vector(1,sm.nnz()) );
			Array<rsb_coo_index_t> JA( dim_vector(1,sm.nnz()) );
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			bool islowtri=true,isupptri=true;
			rsb_flags_t eflags=RSBOI_RF;
			rsb_type_t typecode=RSB_NUMERICAL_TYPE_DOUBLE;
			octave_idx_type nr = sm.rows (); octave_idx_type nc = sm.cols ();

#if 0
			if(nnz==0)/* FIXME: this branch is temporary */
			{
			for (octave_idx_type j = 0; j < nc; j++)
			{
				for (octave_idx_type k = sm.cidx(j); k < sm.cidx(j+1); k++)
				{
					octave_idx_type i=sm.ridx(k);
					IA(k)=i;
					JA(k)=j;
					++nnz;
					if(i>j)isupptri=false;
					else if(i<j)islowtri=false;
				}
			}
			if(isupptri) RSB_DO_FLAG_ADD(eflags,RSB_FLAG_UPPER_TRIANGULAR);
			if(islowtri) RSB_DO_FLAG_ADD(eflags,RSB_FLAG_LOWER_TRIANGULAR);

			if(!(this->A=rsb_allocate_rsb_sparse_matrix_const(sm.data(), (rsb_coo_index_t*)IA.data(), (rsb_coo_index_t*)JA.data(), nnz,typecode, nr, nc, RSBOI_RB, RSBOI_CB, eflags,&errval)))
				RSBOI_ERROR(RSBOI_0_ALLERRMSG);
			}
			else
#endif
			{
			if(!(this->A=rsb_allocate_rsb_sparse_matrix_from_csc_const(sm.data(),sm.ridx(),sm.cidx(), nnz=sm.nnz(),typecode, nr, nc, RSBOI_RB, RSBOI_CB, eflags,&errval)))
				RSBOI_ERROR(RSBOI_0_ALLERRMSG);
			}
			RSBOI_PERROR(errval);
			if(!this->A)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparse_rsb_matrix (const Matrix &m) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			SparseMatrix sm(m);
			rsboi_allocate_rsb_matrix_from_csc_copy(sm);
		}

#if RSBOI_WANT_DOUBLE_COMPLEX
		void rsboi_allocate_rsb_matrix_from_csc_copy(const SparseComplexMatrix &sm)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			octave_idx_type nr = sm.rows ();
			octave_idx_type nc = sm.cols ();
			octave_idx_type nnz=0;
			Array<rsb_coo_index_t> IA( dim_vector(1,sm.nnz()) );
			Array<rsb_coo_index_t> JA( dim_vector(1,sm.nnz()) );
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			bool islowtri=true,isupptri=true;
			rsb_flags_t eflags=RSBOI_RF;
			rsb_type_t typecode=RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX;
#if 0
			for (octave_idx_type j = 0; j < nc; j++)
			{
				for (octave_idx_type k = sm.cidx(j); k < sm.cidx(j+1); k++)
				{
					octave_idx_type i=sm.ridx(k);
					IA(k)=i;
					JA(k)=j;
					++nnz;
					if(i>j)isupptri=false;
					else if(i<j)islowtri=false;
				}
			}
			if(isupptri) RSB_DO_FLAG_ADD(eflags,RSB_FLAG_UPPER_TRIANGULAR);
			if(islowtri) RSB_DO_FLAG_ADD(eflags,RSB_FLAG_LOWER_TRIANGULAR);

			if(!(this->A=rsb_allocate_rsb_sparse_matrix_const(sm.data(), (rsb_coo_index_t*)IA.data(), (rsb_coo_index_t*)JA.data(), nnz,typecode, nr, nc, RSBOI_RB, RSBOI_CB, eflags,&errval)))
				RSBOI_ERROR(RSBOI_0_ALLERRMSG);
#else
			if(!(this->A=rsb_allocate_rsb_sparse_matrix_from_csc_const(sm.data(),sm.ridx(),sm.cidx(), nnz=sm.nnz(),typecode, nr, nc, RSBOI_RB, RSBOI_CB, eflags,&errval)))
				RSBOI_ERROR(RSBOI_0_ALLERRMSG);
#endif
			RSBOI_PERROR(errval);
			if(!this->A)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparse_rsb_matrix (const ComplexMatrix &m) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rsboi_allocate_rsb_matrix_from_csc_copy(SparseComplexMatrix(m));
		}

		octave_sparse_rsb_matrix (const SparseComplexMatrix &sm, rsb_type_t typecode=RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rsboi_allocate_rsb_matrix_from_csc_copy(sm);
		}
#endif

		octave_sparse_rsb_matrix (const SparseMatrix &sm, rsb_type_t typecode=RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			rsboi_allocate_rsb_matrix_from_csc_copy(sm);
		}

		octave_sparse_rsb_matrix (struct rsb_matrix_t * matrix) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX), A(matrix)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			if(!this->A)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparse_rsb_matrix (const octave_sparse_rsb_matrix& T) :
		octave_sparse_matrix (T)  { RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG); this->A=rsb_clone(T.A); };
		octave_idx_type length (void) const { return this->nnz(); }
		octave_idx_type nelem (void) const { return this->nnz(); }
		octave_idx_type numel (void) const { return this->nnz(); }
		octave_idx_type nnz (void) const { RSBOI_0_EMCHECK(this->A);return this->A->nnz; }
		dim_vector dims (void) const { return (dim_vector(this->rows(),this->cols())); }
		octave_idx_type dim1 (void) const { return this->rows(); }
		octave_idx_type dim2 (void) const { return this->cols(); }
		octave_idx_type rows (void) const { RSBOI_0_EMCHECK(this->A);return this->A->m; }
		octave_idx_type cols (void) const { RSBOI_0_EMCHECK(this->A);return this->A->k; }
		octave_idx_type columns (void) const { return this->cols(); }
		octave_idx_type nzmax (void) const { return this->nnz(); }
		octave_idx_type capacity (void) const { return this->nnz(); }
		size_t byte_size (void) const { RSBOI_0_EMCHECK(this->A);return rsb_sizeof(this->A); }

		virtual ~octave_sparse_rsb_matrix (void)
		{
			RSBOI_DEBUG_NOTICE("destroying librsb matrix %p\n",this->A);
			RSBOI_DESTROY(this->A);
		};

		virtual octave_base_value *clone (void) const
		{
			RSBOI_DEBUG_NOTICE("cloning librsb matrix %p\n",this->A);
			return new octave_sparse_rsb_matrix (*this);
		}

		virtual octave_base_value *empty_clone (void) const
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			return new octave_sparse_rsb_matrix ();
		}

#if 0
		octave_value do_index_op(const octave_value_list& idx, bool resize_ok)
		{
			...
#endif

		virtual SparseMatrix sparse_matrix_value(bool = false)const
		{
			struct rsb_coo_matrix_t coo;
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			rsb_nnz_index_t nnz,nzi;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			RSBOI_0_EMCHECK(this->A);
			nnz=this->nnz();
			Array<octave_idx_type> IA( dim_vector(1,nnz) );
			Array<octave_idx_type> JA( dim_vector(1,nnz) );
			Array<RSBOI_T> VA( dim_vector(1,nnz) );
			coo.IA=(rsb_coo_index_t*)IA.data(),coo.JA=(rsb_coo_index_t*)JA.data();
			if(!this->is_real_type())
			{
				Array<Complex> VAC( dim_vector(1,nnz) );
				coo.VA=(RSBOI_T*)VAC.data();
				errval=rsb_get_coo(this->A,coo.VA,coo.IA,coo.JA,RSB_FLAG_C_INDICES_INTERFACE);
				for(nzi=0;nzi<nnz;++nzi)
					((RSBOI_T*)VA.data())[nzi]=((RSBOI_T*)coo.VA)[2*nzi];
			}
			else
			{
				coo.VA=(RSBOI_T*)VA.data();
				errval=rsb_get_coo(this->A,coo.VA,coo.IA,coo.JA,RSB_FLAG_C_INDICES_INTERFACE);
			}
			coo.m=this->rows();
			coo.k=this->cols();
			return SparseMatrix(VA,IA,JA,coo.m,coo.k);
		}

		virtual Matrix matrix_value(bool = false)const
		{
			/* FIXME: inefficient */
			Matrix cm=sparse_matrix_value().matrix_value();
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			return cm;
		}

		virtual octave_value full_value(void)const
		{
			/* FIXME: inefficient */
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			if(this->is_real_type())
				return matrix_value();
			else
				return complex_matrix_value();
		}

#if RSBOI_WANT_DOUBLE_COMPLEX
		virtual ComplexMatrix complex_matrix_value(bool = false)const
		{
			/* FIXME: inefficient */
			octave_sparse_complex_matrix ocm=sparse_complex_matrix_value();
			ComplexMatrix cm=ocm.complex_matrix_value();
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			return cm;
		}

		virtual SparseComplexMatrix sparse_complex_matrix_value(bool = false)const
		{
			struct rsb_coo_matrix_t coo;
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			rsb_nnz_index_t nnz,nzi;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			RSBOI_0_EMCHECK(this->A);
			nnz=this->nnz();
			Array<octave_idx_type> IA( dim_vector(1,nnz) );
			Array<octave_idx_type> JA( dim_vector(1,nnz) );
			Array<Complex> VA( dim_vector(1,nnz) );
			coo.IA=(rsb_coo_index_t*)IA.data(),coo.JA=(rsb_coo_index_t*)JA.data();
			coo.VA=(RSBOI_T*)VA.data();
			errval=rsb_get_coo(this->A,coo.VA,coo.IA,coo.JA,RSB_FLAG_C_INDICES_INTERFACE);
			if(this->is_real_type())
				for(nzi=0;nzi<nnz;++nzi)
					((RSBOI_T*)VA.data())[2*(nnz-1-nzi)+0]=((RSBOI_T*)VA.data())[(nnz-1-nzi)+0],
					((RSBOI_T*)VA.data())[2*(nnz-1-nzi)+1]=0;
			coo.m=this->rows();
			coo.k=this->cols();
			return SparseComplexMatrix(VA,IA,JA,coo.m,coo.k);
		}
#endif

#if RSBOI_WANT_SUBSREF
		octave_value subsref (const std::string &type, const std::list<octave_value_list>& idx)
		{
			octave_value retval;
			int skip = 1;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);

			switch (type[0])
			{
				case '(':
				if (type.length () == 1)
				{
  					octave_idx_type n_idx = idx.front().length ();
					if (n_idx == 2 )
	  				{
	    					idx_vector i = idx.front() (0).index_vector ();
	    					if (! error_state)
	      					{
							if(is_real_type())
							{
								idx_vector j = idx.front() (1).index_vector ();
								RSBOI_T rv;
						  		octave_idx_type ii=-1,jj=-1;
								rsb_err_t errval=RSB_ERR_NO_ERROR;
  								ii=i(0); jj=j(0);
								RSBOI_DEBUG_NOTICE("get_elements (%d %d)\n",ii,jj);
       								errval=rsb_get_elements(this->A,&rv,&ii,&jj,1,RSBOI_NF);
								retval=rv;
								if (! error_state)
								  ;
							}
							else
							{
								idx_vector j = idx.front() (1).index_vector ();
								Complex rv;
						  		octave_idx_type ii=-1,jj=-1;
								rsb_err_t errval=RSB_ERR_NO_ERROR;
  								ii=i(0); jj=j(0);
								RSBOI_DEBUG_NOTICE("get_elements (%d %d) complex\n",ii,jj);
       								errval=rsb_get_elements(this->A,&rv,&ii,&jj,1,RSBOI_NF);
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
		bool is_real_type (void) const { RSBOI_0_EMCHECK(this->A); RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);return this->A->typecode==RSB_NUMERICAL_TYPE_DOUBLE?true:false; }
		bool is_complex_type (void) const { RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG); return !is_real_type(); }
		bool is_bool_type (void) const { return false; }
		bool is_integer_type (void) const { return false; }
		bool is_square (void) const { return this->rows()==this->cols(); }
		bool is_empty (void) const { return false; }
//		int is_struct (void) const { return false; }

		octave_value subsasgn (const std::string& type, const std::list<octave_value_list>& idx, const octave_value& rhs)
		{
			octave_value retval;
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
								if (! error_state)
								{
									if(is_real_type())
									{
										rsb_err_t errval=RSB_ERR_NO_ERROR;
										idx_vector j = idx.front() (1).index_vector ();
										octave_idx_type ii=-1,jj=-1;
										RSBOI_T rv=rhs.double_value();
										ii=i(0); jj=j(0);
										RSBOI_DEBUG_NOTICE("update elements (%d %d)\n",ii,jj);
										errval=rsb_update_elements(this->A,&rv,&ii,&jj,1,RSBOI_NF);
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
										idx_vector j = idx.front() (1).index_vector ();
										octave_idx_type ii=-1,jj=-1;
										Complex rv=rhs.complex_value();
										ii=i(0); jj=j(0);
										RSBOI_DEBUG_NOTICE("update elements (%d %d) complex\n",ii,jj);
										errval=rsb_update_elements(this->A,&rv,&ii,&jj,1,RSBOI_NF);
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
skipimpl:
			return retval;
		}

		octave_base_value *try_narrowing_conversion (void)
		{
			octave_base_value *retval = 0;
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
			return retval;
		}

		//type_conv_fcn numeric_conversion_function (void) const
		type_conv_info numeric_conversion_function (void) const
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			return default_numeric_conversion_function;
		}

	#if 0
		bool isupper (void) const { return false; /**/ }
		bool islower (void) const { return false; /**/ }

		void assign (const octave_value_list& idx, const Matrix& rhs)
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
			std::cerr << "octave_sparse_matrix::assign(idx, rhs);\n";
			//octave_sparse_matrix::assign(idx, rhs);
		}

		octave_sparse_rsb_matrix transpose (void) const
		{
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
			return octave_sparse_rsb_matrix();
		}
	#endif

		void print (std::ostream& os, bool pr_as_read_syntax = false) const
		{
			/* FIXME: what to do with pr_as_read_syntax ? */
			struct rsb_coo_matrix_t coo;
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			rsb_nnz_index_t nnz=this->nnz(),nzi;
			bool ic=this->is_real_type()?false:true;
			Array<octave_idx_type> IA( dim_vector(1,nnz) );
			Array<octave_idx_type> JA( dim_vector(1,nnz) );
			Array<RSBOI_T> VA( dim_vector(1,(ic?2:1)*nnz) );
			std::string c=ic?"complex":"real";
			RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
			coo.VA=(RSBOI_T*)VA.data(),coo.IA=(rsb_coo_index_t*)IA.data(),coo.JA=(rsb_coo_index_t*)JA.data();
			if(coo.VA==NULL)
				nnz=0;
			else
				errval=rsb_get_coo(this->A,coo.VA,coo.IA,coo.JA,RSB_FLAG_C_INDICES_INTERFACE);
			coo.m=this->rows();
			coo.k=this->cols();
			octave_stdout<<RSBOI_FSTR<< "  (rows = "<<coo.m<<", cols = "<<coo.k<<", nnz = "<<nnz<<" ["<<100.0*(((RSBOI_T)nnz)/((RSBOI_T)coo.m))/coo.k<<"%], "<<c<<")\n";
			if(ic)
			for(nzi=0;nzi<nnz;++nzi)
				octave_stdout<<"  ("<<1+IA(nzi)<<", "<<1+JA(nzi)<<") -> "<<((RSBOI_T*)coo.VA)[2*nzi+0]<<" + " <<((RSBOI_T*)coo.VA)[2*nzi+1]<<"i\n";
			else
			for(nzi=0;nzi<nnz;++nzi)
				octave_stdout<<"  ("<<1+IA(nzi)<<", "<<1+JA(nzi)<<") -> "<<((RSBOI_T*)coo.VA)[nzi]<<"\n";
			newline(os);
done:			RSBIO_NULL_STATEMENT_FOR_COMPILER_HAPPINESS
		}

	octave_value diag (octave_idx_type k) const
	{
		octave_value retval;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		RSBOI_0_EMCHECK(this->A);
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
				errval=rsb_getdiag (this->A,(RSBOI_T*)DA.data());
				retval=(DA);
			}
			else
			{
				ComplexMatrix DA(this->rows(),1);
				errval=rsb_getdiag (this->A,(void*)DA.data());
				retval=(DA);
			}
		}
		else
		{
			error(RSBOI_0_NSQERRMSG);
		}
		return retval;
	}

	rsb_err_t rsboi_scale_inv(RSBOI_T alpha)
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		errval=rsb_elemental_scale_inv(this->A,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

	octave_value rsboi_get_scaled_copy_inv(const RSBOI_T alpha)const
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		octave_sparse_rsb_matrix * m = NULL;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		m = new octave_sparse_rsb_matrix(*this);
		if(!m)return m;
		errval=m->rsboi_scale_inv(alpha);
		RSBOI_PERROR(errval);
		return m;
	}

	rsb_err_t rsboi_scale(RSBOI_T alpha)
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		errval=rsb_elemental_scale(this->A,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

	rsb_err_t rsboi_scale(Complex alpha)
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
		errval=rsb_elemental_scale(this->A,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

	octave_value rsboi_get_scaled_copy(const RSBOI_T alpha)const
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		octave_sparse_rsb_matrix * m = NULL;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
#if 0
		m = new octave_sparse_rsb_matrix(*this);
		if(!m)return m;
		errval=m->rsboi_scale(alpha);
		RSBOI_PERROR(errval);
#else
		if(is_real_type())
		m = new octave_sparse_rsb_matrix( rsb_clone_transformed(this->A,RSB_TRANSPOSITION_N,&alpha,RSB_NUMERICAL_TYPE_DOUBLE) );
		else
#if RSBOI_WANT_DOUBLE_COMPLEX
		{Complex calpha;calpha+=alpha;
		m = new octave_sparse_rsb_matrix( rsb_clone_transformed(this->A,RSB_TRANSPOSITION_N,&calpha,RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX) );
		}
#else
		{RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);}
#endif
#endif
		return m;
	}

#if RSBOI_WANT_DOUBLE_COMPLEX
	octave_value rsboi_get_scaled_copy(const Complex alpha)const
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		octave_sparse_rsb_matrix * m = NULL;
		RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
#if 0
		m = new octave_sparse_rsb_matrix(*this);
		if(!m)return m;
		errval=m->rsboi_scale(alpha);
		RSBOI_PERROR(errval);
#else
		m = new octave_sparse_rsb_matrix( rsb_clone_transformed(this->A,RSB_TRANSPOSITION_N,&alpha,RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX) );
#endif
		return m;
	}
#endif

	private:

		DECLARE_OCTAVE_ALLOCATOR
			DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};/* end of class octave_sparse_rsb_matrix definition  */

#if 0
octave_value_list find_nonzero_elem_idx (const class octave_sparse_rsb_matrix & nda, int nargout, octave_idx_type n_to_find, int direction)
{
	// useless
	octave_value retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return retval;
}
#endif

static octave_base_value * default_numeric_conversion_function (const octave_base_value& a)
{
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_CONV_ARG (const octave_sparse_rsb_matrix&);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_WARN(RSBOI_0_UNFFEMSG);

	//octave_base_value ovb=try_narrowing_conversion(void);
	//return new octave_sparse_matrix (v.array_value ());
	/* FIXME: should use conversion to sparse, here, without using dense */
	#if 0
	Matrix m( v.dims() );
	m.fill(0);
	errval|=rsb_matrix_add_to_dense(v.A,&rsboi_one,RSB_DEFAULT_TRANSPOSITION,(RSBOI_T*)m.data(),v.rows(),v.rows(),v.cols(),RSB_BOOL_TRUE);
	RSBOI_PERROR(errval);
	return new octave_sparse_matrix (m);
	#else
	//IA+=1; JA+=1;
	if(v.is_real_type())
		return new octave_sparse_matrix (v.sparse_matrix_value());
	else
		return new octave_sparse_complex_matrix (v.sparse_complex_matrix_value());
	#endif
}

DEFINE_OCTAVE_ALLOCATOR (octave_sparse_rsb_matrix);
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA (octave_sparse_rsb_matrix,
RSB_OI_TYPEINFO_STRING,
RSB_OI_TYPEINFO_TYPE);

DEFCONV (octave_triangular_conv, octave_sparse_rsb_matrix, matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	CAST_CONV_ARG (const octave_sparse_rsb_matrix &);
	return new octave_sparse_matrix (v.matrix_value ());
}

#if 0
DEFCONV (octave_sparse_rsb_to_octave_sparse_conv, sparse_rsb_matrix, sparse_matrix)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_CONV_ARG (const octave_sparse_rsb_matrix &);
	return new octave_sparse_matrix (v.matrix_value ());
}
#endif

DEFUNOP (uplus, sparse_rsb_matrix)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparse_rsb_matrix&);
	return new octave_sparse_rsb_matrix (v);
}

#if 0
DEFUNOP (op_incr, sparse_rsb_matrix)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparse_rsb_matrix&);
	const octave_idx_type rn=v.A->m,cn=v.A->k;
	Matrix v2(rn,cn);
	octave_value retval=v2;
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	errval|=rsb_matrix_add_to_dense(v.A,&rsboi_one,RSB_DEFAULT_TRANSPOSITION,(RSBOI_T*)v2.data(),rn,rn,cn,RSB_BOOL_TRUE);
	//v=octave_ma(idx, v2.matrix_value());
	return v2;
}

DEFUNOP (op_decr, sparse_rsb_matrix)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparse_rsb_matrix&);
	const octave_idx_type rn=v.A->m,cn=v.A->k;
	Matrix v2(rn,cn);
	octave_value retval=v2;
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	errval|=rsb_matrix_add_to_dense(v.A,&rsboi_one,RSB_DEFAULT_TRANSPOSITION,(RSBOI_T*)v2.data(),rn,rn,cn,RSB_BOOL_TRUE);
	//v=octave_ma(idx, v2.matrix_value());
	return v2;
}
#endif

DEFUNOP (uminus, sparse_rsb_matrix)
{
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparse_rsb_matrix&);
	octave_sparse_rsb_matrix * m = new octave_sparse_rsb_matrix(v);
	if(!m)return m;
	rsb_negation(m->A);
	return m;
}

DEFUNOP (transpose, sparse_rsb_matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparse_rsb_matrix&);
	octave_sparse_rsb_matrix * m = new octave_sparse_rsb_matrix(v);
	RSBOI_TODO("here, the best solution would be to use some get_transposed() function");
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	if(!m)return m;
	errval=rsb_transpose(&m->A);
	RSBOI_PERROR(errval);
	return m;
}

DEFUNOP (htranspose, sparse_rsb_matrix)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	CAST_UNOP_ARG (const octave_sparse_rsb_matrix&);
	octave_sparse_rsb_matrix * m = new octave_sparse_rsb_matrix(v);
	RSBOI_TODO("here, the best solution would be to use some get_transposed() function");
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	if(!m)return m;
	errval=rsb_htranspose(&m->A);
	RSBOI_PERROR(errval);
	return m;
}


octave_value rsboi_spsv(const octave_sparse_rsb_matrix&v1, const octave_matrix&v2,rsb_trans_t transa)
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
	errval=rsb_spsm(transa,&rsboi_one,v1.A,nrhs,RSB_OI_DMTXORDER,&rsboi_zero,(const RSBOI_T*)retval.data(),ldb,(RSBOI_T*)retval.data(),ldc);
	if(RSBOI_SOME_ERROR(errval))
	{
		if(errval==RSB_ERR_INVALID_NUMERICAL_DATA)
			RSBOI_PERROR(errval);// FIXME: need a specific error message here
		else
			RSBOI_PERROR(errval);// FIXME: generic case, here
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
	errval=rsb_spsm(transa,&rsboi_one,v1.A,nrhs,RSB_OI_DMTXORDER,&rsboi_zero,(const RSBOI_T*)retval.data(),ldb,(RSBOI_T*)retval.data(),ldc);
	if(RSBOI_SOME_ERROR(errval))
	{
		if(errval==RSB_ERR_INVALID_NUMERICAL_DATA)
			RSBOI_PERROR(errval);// FIXME: need a specific error message here
		else
			RSBOI_PERROR(errval);// FIXME: generic case, here
		for(octave_idx_type i=0;i<nels;++i)
			retval(i)=octave_NaN;
	}
	return retval;
	}
}

DEFBINOP(ldiv, sparse_rsb_matrix, matrix)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_matrix&);
	return rsboi_spsv(v1,v2,RSB_TRANSPOSITION_N);
}

DEFBINOP(trans_ldiv, sparse_rsb_matrix, matrix)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_matrix&);
	return rsboi_spsv(v1,v2,RSB_TRANSPOSITION_T);
}

DEFBINOP(el_div, sparse_rsb_matrix, matrix)
{
	Matrix retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return retval;
}

DEFBINOP(el_ldiv, sparse_rsb_matrix, matrix)
{
	Matrix retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return retval;
}

DEFBINOP(div, sparse_rsb_matrix, matrix)
{
	Matrix retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return retval;
}

DEFBINOP(rsb_s_div, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.scalar_value());
}

DEFBINOP(rsb_s_mul, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy(v2.scalar_value());
}

#if 0
DEFBINOP(rsb_s_pow, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_power_copy(v2.scalar_value());
}
#endif

DEFASSIGNOP (assign, sparse_rsb_matrix, sparse_rsb_matrix)
{
	/* FIXME : I dunno how to trigger this! */
	CAST_BINOP_ARGS (octave_sparse_rsb_matrix &, const octave_sparse_rsb_matrix&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	rsb_assign(v1.A, v2.A);
	return octave_value();
}

DEFASSIGNOP (assignm, sparse_rsb_matrix, matrix)
{
	CAST_BINOP_ARGS (octave_sparse_rsb_matrix &, const octave_matrix&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_DESTROY(v1.A);
	octave_value retval;
	//v1.assign(idx, v2.matrix_value());
	v1=(idx, v2.matrix_value());
	//retval=v1;
	retval=v2.matrix_value();
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return retval;
}

#if 0
DEFASSIGNOP(rsb_op_mul_eq_s, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (octave_sparse_rsb_matrix &, const octave_scalar&);
	octave_value retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_PERROR(v1.rsboi_scale(v2.scalar_value()));
	retval=v1.matrix_value();
	return retval;
}

DEFASSIGNOP(rsb_op_div_eq_s, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (octave_sparse_rsb_matrix &, const octave_scalar&);
	octave_value retval;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	RSBOI_PERROR(v1.rsboi_scale_inv(v2.scalar_value()));
	retval=v1.matrix_value();
	return retval;
}
#endif

DEFBINOP(rsb_el_mul_s, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy(v2.scalar_value());
}

DEFBINOP(rsb_el_div_s, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	return v1.rsboi_get_scaled_copy_inv(v2.scalar_value());
}

DEFBINOP(el_pow, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	octave_sparse_rsb_matrix * m = new octave_sparse_rsb_matrix(v1);
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	RSBOI_T alpha=v2.scalar_value();
	if(!m)return m;
	errval=rsb_elemental_pow(m->A,&alpha);
	RSBOI_PERROR(errval);
	return m;
}

#ifdef RSB_FULLY_IMPLEMENTED
DEFASSIGNOP (assigns, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	v1.assign(idx, v2.matrix_value());
	RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	return octave_value();
}
#endif

DEFBINOP(op_sub, sparse_rsb_matrix, sparse_rsb_matrix)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_sparse_rsb_matrix&);
	octave_sparse_rsb_matrix*sm = new octave_sparse_rsb_matrix();
	octave_value retval = sm;
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	/* FIXME */
	sm->A=rsb_matrix_sum(RSB_TRANSPOSITION_N,&rsboi_one,v1.A,RSB_TRANSPOSITION_N,&rsboi_mone,v2.A,&errval);
	RSBOI_PERROR(errval);
	if(!sm->A)
		RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
	return retval;
}

DEFBINOP(op_add, sparse_rsb_matrix, sparse_rsb_matrix)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_sparse_rsb_matrix&);
	octave_sparse_rsb_matrix*sm = new octave_sparse_rsb_matrix();
	octave_value retval = sm;
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	sm->A=rsb_matrix_sum(RSB_TRANSPOSITION_N,&rsboi_one,v1.A,RSB_TRANSPOSITION_N,&rsboi_one,v2.A,&errval);
	RSBOI_PERROR(errval);
	if(!sm->A)
		RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
	return retval;
}

DEFBINOP(op_spmul, sparse_rsb_matrix, sparse_rsb_matrix)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_sparse_rsb_matrix&);
	octave_sparse_rsb_matrix*sm = new octave_sparse_rsb_matrix();
	octave_value retval = sm;
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	/* FIXME: what if they are not both of the same type ? it would be nice to have a conversion.. */
	sm->A=rsb_matrix_mul(RSB_TRANSPOSITION_N,&rsboi_one,v1.A,RSB_TRANSPOSITION_N,&rsboi_one,v2.A,&errval);
	RSBOI_PERROR(errval);
	if(!sm->A)
		RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
	return retval;
}

DEFBINOP(op_mul, sparse_rsb_matrix, matrix)
{
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_matrix&);
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	if(v1.is_real_type())
	{
		const Matrix b = v2.matrix_value ();
		octave_idx_type b_nc = b.cols ();
		octave_idx_type b_nr = b.rows ();
		octave_idx_type ldb=b_nr;
		octave_idx_type ldc=v1.rows();
		octave_idx_type nrhs=b_nc;
		Matrix retval(ldc,nrhs,RSBOI_ZERO);
		if(v1.columns()!=b_nr) { error("matrices dimensions do not match!\n"); return Matrix(); }
		errval=rsb_spmm(RSB_TRANSPOSITION_N,&rsboi_one,v1.A,nrhs,RSB_OI_DMTXORDER,(RSBOI_T*)b.data(),ldb,&rsboi_zero,(RSBOI_T*)retval.data(),ldc);
		RSBOI_PERROR(errval);
		return retval;
	}
	else
	{
		const ComplexMatrix b = v2.complex_matrix_value ();
		octave_idx_type b_nc = b.cols ();
		octave_idx_type b_nr = b.rows ();
		octave_idx_type ldb=b_nr;
		octave_idx_type ldc=v1.rows();
		octave_idx_type nrhs=b_nc;
		ComplexMatrix retval(ldc,nrhs,RSBOI_ZERO);
		if(v1.columns()!=b_nr) { error("matrices dimensions do not match!\n"); return ComplexMatrix(); }
		errval=rsb_spmm(RSB_TRANSPOSITION_N,&rsboi_one,v1.A,nrhs,RSB_OI_DMTXORDER,(RSBOI_T*)b.data(),ldb,&rsboi_zero,(RSBOI_T*)retval.data(),ldc);
		RSBOI_PERROR(errval);
		return retval;
	}
}

DEFBINOP(op_trans_mul, sparse_rsb_matrix, matrix)
{
	// ".'*"  operator
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_matrix&);
	if(v1.is_real_type())
	{
		const Matrix b = v2.matrix_value ();
		octave_idx_type b_nc = b.cols ();
		octave_idx_type b_nr = b.rows ();
		octave_idx_type ldb=b_nr;
		octave_idx_type ldc=v1.columns();
		octave_idx_type nrhs=b_nc;
		Matrix retval(ldc,nrhs,RSBOI_ZERO);
		if(v1.rows()!=b_nr) { error("matrices dimensions do not match!\n"); return Matrix(); }
		//octave_stdout << "have: ldc=" <<ldc << " bc=" << b_nc<< " nrhs=" << nrhs << " retval="<< retval<< "\n";

		//errval=rsb_spmv(RSB_TRANSPOSITION_T,&rsboi_one,v1.A,(RSBOI_T*)b.data(),RSBOI_OV_STRIDE,&rsboi_one,(RSBOI_T*)retval.data(),RSBOI_OV_STRIDE);
		errval=rsb_spmm(RSB_TRANSPOSITION_T,&rsboi_one,v1.A,nrhs,RSB_OI_DMTXORDER,(RSBOI_T*)b.data(),ldb,&rsboi_zero,(RSBOI_T*)retval.data(),ldc);
		RSBOI_PERROR(errval);
		return retval;
	}
	else
	{
		const ComplexMatrix b = v2.complex_matrix_value ();
		octave_idx_type b_nc = b.cols ();
		octave_idx_type b_nr = b.rows ();
		octave_idx_type ldb=b_nr;
		octave_idx_type ldc=v1.columns();
		octave_idx_type nrhs=b_nc;
		ComplexMatrix retval(ldc,nrhs,RSBOI_ZERO);
		if(v1.rows()!=b_nr) { error("matrices dimensions do not match!\n"); return ComplexMatrix(); }
		//octave_stdout << "have: ldc=" <<ldc << " bc=" << b_nc<< " nrhs=" << nrhs << " retval="<< retval<< "\n";

		//errval=rsb_spmv(RSB_TRANSPOSITION_T,&rsboi_one,v1.A,(RSBOI_T*)b.data(),RSBOI_OV_STRIDE,&rsboi_one,(RSBOI_T*)retval.data(),RSBOI_OV_STRIDE);
		errval=rsb_spmm(RSB_TRANSPOSITION_T,&rsboi_one,v1.A,nrhs,RSB_OI_DMTXORDER,(RSBOI_T*)b.data(),ldb,&rsboi_zero,(RSBOI_T*)retval.data(),ldc);
		RSBOI_PERROR(errval);
		return retval;
	}
}

static void install_sparsersb_ops (void)
{
	RSBOI_DEBUG_NOTICE(RSBOI_D_EMPTY_MSG);
	#ifdef RSB_FULLY_IMPLEMENTED
	/* boolean pattern-based not */
	INSTALL_UNOP (op_not, octave_sparse_rsb_matrix, op_not);
	/* to-dense operations */
	INSTALL_ASSIGNOP (op_asn_eq, octave_sparse_rsb_matrix, octave_scalar, assigns);
	/* ? */
	INSTALL_UNOP (op_uplus, octave_sparse_rsb_matrix, uplus);
	/* elemental comparison, evaluate to sparse or dense boolean matrices */
	INSTALL_BINOP (op_eq, octave_sparse_rsb_matrix, , );
	INSTALL_BINOP (op_le, octave_sparse_rsb_matrix, , );
	INSTALL_BINOP (op_lt, octave_sparse_rsb_matrix, , );
	INSTALL_BINOP (op_ge, octave_sparse_rsb_matrix, , );
	INSTALL_BINOP (op_gt, octave_sparse_rsb_matrix, , );
	INSTALL_BINOP (op_ne, octave_sparse_rsb_matrix, , );
	/* pure elemental; scalar and sparse arguments ?! */
								 // ?
	INSTALL_BINOP (op_el_ldiv, octave_sparse_rsb_matrix, , );
	INSTALL_BINOP (op_el_and, octave_sparse_rsb_matrix, , );
	INSTALL_BINOP (op_el_or, octave_sparse_rsb_matrix, , );
	/* shift operations: they may be left out from the implementation */
	INSTALL_BINOP (op_lshift, octave_sparse_rsb_matrix, , );
	INSTALL_BINOP (op_rshift, octave_sparse_rsb_matrix, , );
	#endif
	//INSTALL_WIDENOP (octave_sparse_rsb_matrix, octave_sparse_matrix,octave_sparse_rsb_to_octave_sparse_conv);/* a DEFCONV .. */
	//INSTALL_ASSIGNCONV (octave_sparse_rsb_matrix, octave_sparse_matrix,octave_sparse_matrix);/* .. */
	// no need for the following: need a good conversion function, though
	//INSTALL_UNOP (op_incr, octave_sparse_rsb_matrix, op_incr);
	//INSTALL_UNOP (op_decr, octave_sparse_rsb_matrix, op_decr);
	INSTALL_BINOP (op_el_mul, octave_sparse_rsb_matrix, octave_scalar, rsb_el_mul_s);
//	INSTALL_ASSIGNOP (op_mul_eq, octave_sparse_rsb_matrix, octave_scalar, rsb_op_mul_eq_s); // 20110313 not effective
//	INSTALL_ASSIGNOP (op_div_eq, octave_sparse_rsb_matrix, octave_scalar, rsb_op_div_eq_s); // 20110313 not effective
	INSTALL_BINOP (op_el_div, octave_sparse_rsb_matrix, octave_scalar, rsb_el_div_s);
	INSTALL_BINOP (op_el_pow, octave_sparse_rsb_matrix, octave_scalar, el_pow);
	INSTALL_UNOP (op_uminus, octave_sparse_rsb_matrix, uminus);
	INSTALL_BINOP (op_ldiv, octave_sparse_rsb_matrix, octave_matrix, ldiv);
	INSTALL_BINOP (op_el_ldiv, octave_sparse_rsb_matrix, octave_matrix, el_ldiv);
	INSTALL_BINOP (op_div, octave_sparse_rsb_matrix, octave_matrix, div);
	INSTALL_BINOP (op_div, octave_sparse_rsb_matrix, octave_scalar, rsb_s_div);
	INSTALL_BINOP (op_mul, octave_sparse_rsb_matrix, octave_scalar, rsb_s_mul);
	//INSTALL_BINOP (op_pow, octave_sparse_rsb_matrix, octave_scalar, rsb_s_pow);
	INSTALL_BINOP (op_el_div, octave_sparse_rsb_matrix, octave_matrix, el_div);
	INSTALL_UNOP (op_transpose, octave_sparse_rsb_matrix, transpose);
	INSTALL_UNOP (op_hermitian, octave_sparse_rsb_matrix, htranspose);
	INSTALL_ASSIGNOP (op_asn_eq, octave_sparse_rsb_matrix, octave_sparse_matrix, assign);
	INSTALL_ASSIGNOP (op_asn_eq, octave_sparse_rsb_matrix, octave_matrix, assignm);
	INSTALL_BINOP (op_mul, octave_sparse_rsb_matrix, octave_matrix, op_mul);
	//INSTALL_BINOP (op_pow, octave_sparse_rsb_matrix, octave_matrix, op_pow);
	INSTALL_BINOP (op_sub, octave_sparse_rsb_matrix, octave_sparse_rsb_matrix, op_sub);
	INSTALL_BINOP (op_add, octave_sparse_rsb_matrix, octave_sparse_rsb_matrix, op_add);
	//INSTALL_BINOP (op_trans_add, octave_sparse_rsb_matrix, octave_sparse_rsb_matrix, op_trans_add);
	INSTALL_BINOP (op_mul, octave_sparse_rsb_matrix, octave_sparse_rsb_matrix, op_spmul);
	INSTALL_BINOP (op_trans_mul, octave_sparse_rsb_matrix, octave_matrix, op_trans_mul);
	INSTALL_BINOP (op_trans_ldiv, octave_sparse_rsb_matrix, octave_matrix, trans_ldiv);
	//INSTALL_BINOP (op_mul_trans, octave_sparse_rsb_matrix, octave_matrix, op_mul_trans);
	//INSTALL_BINOP (op_mul_trans, octave_sparse_rsb_matrix, octave_matrix, op_mul_trans);
	//INSTALL_BINOP (op_herm_mul, octave_sparse_rsb_matrix, octave_matrix, op_herm_mul);
	//INSTALL_BINOP (op_mul_herm, octave_sparse_rsb_matrix, octave_matrix, op_mul_herm);
	//INSTALL_BINOP (op_el_not_and, octave_sparse_rsb_matrix, octave_matrix, op_el_not_and);
	//INSTALL_BINOP (op_el_not_or , octave_sparse_rsb_matrix, octave_matrix, op_el_not_or );
	//INSTALL_BINOP (op_el_and_not, octave_sparse_rsb_matrix, octave_matrix, op_el_and_not);
	//INSTALL_BINOP (op_el_or _not, octave_sparse_rsb_matrix, octave_matrix, op_el_or _not);
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
		if(RSBOI_SOME_ERROR(errval=rsb_init(RSB_NULL_INIT_OPTIONS)))
		{
			RSBOI_PERROR(errval);//FIXME: temporary
			RSBOI_ERROR("");
			goto err;
		}
		rsboi_initialized=true;
	}
	else
		;/* already initialized */

	if (!rsboi_sparse_rsb_loaded)
	{
		octave_sparse_rsb_matrix::register_type ();
		install_sparsersb_ops ();
		rsboi_sparse_rsb_loaded=true;
		mlock();
	}
	return;
	err:
	RSBIO_NULL_STATEMENT_FOR_COMPILER_HAPPINESS
}

// PKG_ADD: autoload (RSBOI_FNS, RSBOI_FNS".oct");
DEFUN_DLD (RSB_SPARSERSB_LABEL, args, ,
"-*- texinfo -*-\n\
@deftypefn {Loadable Function} {@var{s} =} "RSBOI_FNS" (@var{a})\n\
Create a sparse RSB matrix from the full matrix @var{a}.\n"\
/*is forced back to a full matrix if resulting matrix is sparse\n*/\
"\n\
@deftypefnx {Loadable Function} {@var{s} =} "RSBOI_FNS" (@var{filename})\n\
Create a sparse RSB matrix by loading the Matrix Market matrix file named @var{filename}.\n"\
"In the case @var{filename} is \""RSBOI_LIS"\", a string listing the available numerical types with BLAS-style characters will be returned.\n"\
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
Equivalent to @code{"RSBOI_FNS" ([], [], [], @var{m}, @var{n}, 0)}\n\
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
	octave_sparse_rsb_matrix*matrix=NULL;
	bool ic0=nargin>0?(args(0).is_complex_type()):false;
	bool ic3=nargin>2?(args(2).is_complex_type()):false;

	RSBOI_DEBUG_NOTICE("in sparsersb()\n");

	//if(ic3 || ic0)
	if(ic0)
	{
		RSBOI_WARN(RSBOI_O_MISSIMPERRMSG);
	}

	if(ic3 || ic0)
#if RSBOI_WANT_DOUBLE_COMPLEX
		RSBOI_WARN(RSBOI_0_UNCFEMSG);
#else
		RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);
#endif
	install_sparse_rsb();
	if (nargin == 1 || nargin == 2)
	{
		rsb_type_t typecode=RSBOI_TYPECODE;
		if (nargin >= 2)
#if RSBOI_WANT_DOUBLE_COMPLEX
			typecode=RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX;
#else
			RSBOI_0_ERROR(RSBOI_0_NOCOERRMSG);
#endif
		if(args(0).is_sparse_type())
		{
			if(args(0).type_name()==RSB_OI_TYPEINFO_STRING)
			{
				RSBOI_WARN(RSBOI_0_UNFFEMSG);
				retval.append(matrix=(octave_sparse_rsb_matrix*)(args(0).get_rep()).clone());
			}
			else
			{
				if(!ic0)
				{
					const SparseMatrix m = args(0).sparse_matrix_value();
					if (error_state) goto err;
					retval.append(matrix=new octave_sparse_rsb_matrix(m,typecode));
				}
#if RSBOI_WANT_DOUBLE_COMPLEX
				else
				{
					const SparseComplexMatrix m = args(0).sparse_complex_matrix_value();
					if (error_state) goto err;
					retval.append(matrix=new octave_sparse_rsb_matrix(m,typecode));
				}
#endif
			}
		}
		else
		if(args(0).is_string())
		{
			const std::string m = args(0).string_value();
			if (error_state) goto err;
			if(m==RSBOI_LIS)
			{
				retval.append(RSB_NUMERICAL_TYPE_PREPROCESSOR_SYMBOLS);
				goto ret;
			}
			else
			{
				RSBOI_WARN(RSBOI_0_UNFFEMSG);
				RSBOI_WARN("shall set the type, here");
				retval.append(matrix=new octave_sparse_rsb_matrix(m));
			}
		}
		else
		{
			if(!ic0)
			{
				Matrix m = args(0).matrix_value();
				if (error_state) goto err;
				retval.append(matrix=new octave_sparse_rsb_matrix(m));
			}
#if RSBOI_WANT_DOUBLE_COMPLEX
			else
			{
				ComplexMatrix m = args(0).complex_matrix_value();
				if (error_state) goto err;
				retval.append(matrix=new octave_sparse_rsb_matrix(m));
			}
#endif
		}
	}
	else
	if (nargin >= 3 && nargin <= 6)
	{
		rsb_flags_t eflags=RSBOI_DCF;
		octave_idx_type nr=0,nc=0;
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
			nr=args(3).scalar_value();/* FIXME: need index value here! */
			nc=args(4).scalar_value();
			if(nr<=0 || nc<=0)
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
#if RSBOI_WANT_IDX_VECTOR_CONST
			idx_vector iv=args(0).index_vector ();
			idx_vector jv=args(1).index_vector ();
			retval.append(matrix=new octave_sparse_rsb_matrix( iv, jv, args(2).matrix_value(),nr,nc,eflags ));
#else
			retval.append(matrix=new octave_sparse_rsb_matrix( args(0).matrix_value(), args(1).matrix_value(), args(2).matrix_value(),nr,nc,eflags ));
#endif
		}

#if RSBOI_WANT_DOUBLE_COMPLEX
		else
		{
#if RSBOI_WANT_IDX_VECTOR_CONST
			idx_vector iv=args(0).index_vector ();
			idx_vector jv=args(1).index_vector ();
			retval.append(matrix=new octave_sparse_rsb_matrix( iv, jv, args(2).complex_matrix_value(),nr,nc,eflags ));
#else
			retval.append(matrix=new octave_sparse_rsb_matrix( args(0).matrix_value(), args(1).matrix_value(), args(2).complex_matrix_value(),nr,nc,eflags ));
#endif
		}
#endif
	}
	else
		goto errp;
	if(!matrix)
	{
		RSBOI_WARN(RSBOI_0_NEEDERR);
		RSBOI_DEBUG_NOTICE(RSBOI_0_FATALNBMSG);
	}
#if RSBOI_WANT_HEAVY_DEBUG
	if(!rsb_is_correctly_built_rcsr_matrix(matrix->A)) // non-declared function
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
