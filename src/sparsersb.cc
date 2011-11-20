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
// mkoctfile -g -I../../librsb -o sparsersb.o sparsersb.cc

/*
 * TODO:
 * behaviour with complex data is often undefined. shall fix this.
 * should implement compound_binary_op
 * should use octave_stdout in many situations, instead of librsb's printf statements.
 * for future versions, see http://www.gnu.org/software/octave/doc/interpreter/index.html#Top
 * for thorough testing, see Octave's test/build_sparse_tests.sh
 * need a specialized error dumping routine 
 *
 * NOTES:
 * 20110312 why isstruct() gives 1 ? this invalidates tril, triu
 * 20110312 should unify the constructors into one
 * octave_sparse_matrix in ../src/ov-re-sparse.h
 * SparseMatrix in dSparse.h
 * should issue error if rsboi_zero or other constants get overridden
 * should avoid copying of temporary vectors in spsv/spmv ops
 * should replace the use of rsb_perror with rsboi_perror
 * should print some memory usage statistics and optimize memory usage
 * see also OPERATORS/op-sm-sm.cc */

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
	octave_stdout << "In "<<__func__<<"(), in file "<<__FILE__<<" at line "<<__LINE__<<":\n"<<MSG;
#define RSBOI_DUMP RSBOI_PRINTF
#else
#define RSBOI_DUMP( ... )
#define RSBOI_DEBUG_NOTICE( ... )
#define RSBOI_ERROR( ... )
#endif
#define RSBOI_TYPECODE RSB_NUMERICAL_TYPE_DOUBLE
#define RSBOI_RB RSB_DEFAULT_ROW_BLOCKING
#define RSBOI_CB RSB_DEFAULT_COL_BLOCKING
//#define RSBOI_RF RSB_FLAG_DEFAULT_STORAGE_FLAGS
#define RSBOI_RF RSB_FLAG_DEFAULT_RSB_MATRIX_FLAGS 
#define RSBOI_NF RSB_FLAG_NOFLAGS
#define RSBOI_T double
#define RSBOI_MP(M) RSBOI_DUMP(RSB_PRINTF_MATRIX_SUMMARY_ARGS(M))
#undef RSB_FULLY_IMPLEMENTED
#define RSBOI_DESTROY(OM) {rsb_free_sparse_matrix(OM);(OM)=NULL;}
#define RSBOI_SOME_ERROR(ERRVAL) (ERRVAL)!=RSB_ERR_NO_ERROR
#define RSBOI_0_ERROR error
#define RSBOI_0_ALLERRMSG "error allocating matrix!\n"
#define RSBOI_0_EMERRMSG  "data structure is corrupt (unexpected NULL matrix pointer)!\n"
#define RSBOI_0_EMCHECK(M) if(!(M))RSBOI_0_ERROR(RSBOI_0_EMERRMSG);
#define RSBOI_FNSS(S)	#S
//#define RSBOI_FNS	RSBOI_FNSS(RSB_SPARSERSB_LABEL)
#define RSBOI_FNS	"sparsersb"

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
#define ORSB_RSB_TYPE_FLAG(OBJ) (((OBJ).is_complex_type())?RSB_NUMERICAL_TYPE_DOUBLE:RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX)
#else
#define ORSB_RSB_TYPE_FLAG(OBJ) RSB_NUMERICAL_TYPE_DOUBLE
#endif

#define RSBOI_WANT_SUBSREF 1
#define RSBOI_WANT_HEAVY_DEBUG 0
//#define RSBOI_PERROR(E) rsb_perror(E)
#define RSBOI_PERROR(E) if(RSBOI_SOME_ERROR(E))octave_stdout<<"librsb error:"<<rsb_strerror(E)<<"\n"
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

static bool sparsersb_tester()
{
	if(sizeof(octave_idx_type)!=sizeof(rsb_coo_index_t))
	{
		RSBOI_ERROR("Index sizes of Octave differs from that of RSB:"
			" a conversion is needed, but yet unsupported in this version.");
	//	RSBOI_O_ERROR("Index sizes of Octave differs from that of RSB:"
	//		" a conversion is needed, but yet unsupported in this version.");
		goto err;
	}
	// TODO: THIS FUNCTION IS INCOMPLETE
	return true;
err:
	return false;
}

static bool sparse_rsb_loaded = false; /* a global variable */

class octave_sparse_rsb_matrix : public octave_sparse_matrix
{
	private:
	public:
	struct rsb_matrix_t * A;
	public:
		octave_sparse_rsb_matrix (void) : octave_sparse_matrix(RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE("");
			this->A=NULL;
		};

		octave_sparse_rsb_matrix (const octave_sparse_matrix &sm) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			RSBOI_DEBUG_NOTICE("");
		}

		octave_sparse_rsb_matrix (const std::string &fn, rsb_type_t typecode=RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			RSBOI_DEBUG_NOTICE("");
			if(!(this->A=rsb_load_matrix_file_as_matrix_market(fn.c_str(),RSBOI_RF,typecode,&errval)))
					
				RSBOI_ERROR("error allocating an rsb matrix!\n");
			RSBOI_PERROR(errval);
			if(!this->A)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparse_rsb_matrix (const Matrix &IM, const Matrix &JM, const Matrix &SM, rsb_flags_t eflags=RSB_FLAG_DUPLICATES_SUM) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			/*
			 TODO: shall check if the arrays lenght is the same, and so the types!
			 FIXME: the following copies are not necessary !
			*/
			octave_idx_type nnz=IM.rows()*IM.cols();
			Array<rsb_coo_index_t> IA( dim_vector(1,nnz) );
			Array<rsb_coo_index_t> JA( dim_vector(1,nnz) );
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			bool islowtri=true,isupptri=true;

			RSBOI_DEBUG_NOTICE("\n");
			//RSBOI_DEBUG_NOTICE("oc %d\n",IM.rows()*IM.cols());

			for (octave_idx_type n = 0; n < nnz; n++)
			{
				rsb_coo_index_t i=IM.data()[n]-1,j=JM.data()[n]-1;
				IA(n)=i, JA(n)=j;
				if(i>j)isupptri=false;
				else if(i<j)islowtri=false;
			}
			if(isupptri) RSB_DO_FLAG_ADD(eflags,RSB_FLAG_UPPER_TRIANGULAR);
			if(islowtri) RSB_DO_FLAG_ADD(eflags,RSB_FLAG_LOWER_TRIANGULAR);
				//printf("%d %d %lg\n",IA(n),JA(n),((RSBOI_T*)SM.data())[n]);

			if(!(this->A=rsb_allocate_rsb_sparse_matrix_const(SM.data(), (rsb_coo_index_t*)IA.data(), (rsb_coo_index_t*)JA.data(), nnz, RSBOI_TYPECODE, 0, 0, RSBOI_RB, RSBOI_CB,
				RSBOI_RF|eflags
				,&errval))
				)
				RSBOI_ERROR("error allocating an rsb matrix!\n");
			RSBOI_MP(this->A);
								 // FIXME: need to set symmetry/triangle flags
			//rsb_mark_matrix_with_type_flags(this->A);
			RSBOI_MP(this->A);
			RSBOI_PERROR(errval);
			if(!this->A)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparse_rsb_matrix (const Matrix &m) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			// FIXME: need a specialized contructor in RSB itself
			RSBOI_DEBUG_NOTICE("");
			SparseMatrix sm(m); // FIXME
			octave_idx_type nr = sm.rows ();
			octave_idx_type nc = sm.cols ();
			octave_idx_type nnz=0;
			Array<rsb_coo_index_t> IA( dim_vector(1,sm.nnz()) );
			Array<rsb_coo_index_t> JA( dim_vector(1,sm.nnz()) );
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			bool islowtri=true,isupptri=true;
			rsb_flags_t eflags=RSBOI_RF;
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

			//  	std::cerr << "  rsb    making " << br << " x " << bc << "  matrix.. \n" ;

			//	this->A=rsb_allocate_bcsr_sparse_matrix (
			//		sm.data(), (rsb_coo_index_t*)IA.data(),  (rsb_coo_index_t*)JA.data(), sm.nnz(), RSBOI_TYPECODE , nr, nc, br, bc);

			if(!(this->A=rsb_allocate_rsb_sparse_matrix_const(sm.data(), (rsb_coo_index_t*)IA.data(), (rsb_coo_index_t*)JA.data(), nnz, RSBOI_TYPECODE , nr, nc, RSBOI_RB, RSBOI_CB, eflags,&errval)))
				RSBOI_ERROR(RSBOI_0_ALLERRMSG);
			// FIXME: need to set symmetry/triangle flags
			//rsb_mark_matrix_with_type_flags(this->A);
			RSBOI_PERROR(errval);
			if(!this->A)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparse_rsb_matrix (const SparseMatrix &sm, rsb_type_t typecode=RSBOI_TYPECODE) : octave_sparse_matrix (RSBIO_DEFAULT_CORE_MATRIX)
		{
			// FIXME: need a specialized contructor in RSB itself
			RSBOI_DEBUG_NOTICE("");
			octave_idx_type nr = sm.rows ();
			octave_idx_type nc = sm.cols ();
			octave_idx_type nnz=0;
			Array<rsb_coo_index_t> IA( dim_vector(1,sm.nnz()) );
			Array<rsb_coo_index_t> JA( dim_vector(1,sm.nnz()) );
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			bool islowtri=true,isupptri=true;
			rsb_flags_t eflags=RSBOI_RF;
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

			//  	std::cerr << "  rsb    making " << br << " x " << bc << "  matrix.. \n" ;

			//	this->A=rsb_allocate_bcsr_sparse_matrix (
			//		sm.data(), (rsb_coo_index_t*)IA.data(),  (rsb_coo_index_t*)JA.data(), sm.nnz(), RSBOI_TYPECODE , nr, nc, br, bc);

			if(!(this->A=rsb_allocate_rsb_sparse_matrix_const(((const rsb_byte_t*)sm.data()), (rsb_coo_index_t*)IA.data(), (rsb_coo_index_t*)JA.data(), nnz, typecode , nr, nc, RSBOI_RB, RSBOI_CB, eflags,&errval)))
				RSBOI_ERROR("error allocating an rsb matrix!\n");
								 // FIXME: need to set symmetry/triangle flags
			//rsb_mark_matrix_with_type_flags(this->A);
			RSBOI_PERROR(errval);
			if(!this->A)
				RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
		}

		octave_sparse_rsb_matrix (const octave_sparse_rsb_matrix& T) :
		octave_sparse_matrix (T)  { RSBOI_DEBUG_NOTICE(""); this->A=rsb_clone(T.A); };
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
			// FIXME: I am not sure how and when this works
			RSBOI_DEBUG_NOTICE("cloning librsb matrix %p\n",this->A);
			return new octave_sparse_rsb_matrix (*this);
		}

		virtual octave_base_value *empty_clone (void) const
		{
			RSBOI_DEBUG_NOTICE("");
			return new octave_sparse_rsb_matrix ();
		}

#if 0
		octave_value do_index_op(const octave_value_list& idx, bool resize_ok)
		{
			octave_value retval;
			octave_idx_type n_idx = idx.length ();

  switch (n_idx)
    {
    case 0:
      retval = matrix;
	RSBOI_DEBUG_NOTICE("UNFINISHED\n");
      break;

    case 1:
      {
	RSBOI_DEBUG_NOTICE("UNFINISHED\n");
	idx_vector i = idx (0).index_vector ();

	if (! error_state)
	  retval = octave_value (matrix.index (i, resize_ok));
      }
      break;

    default:
      {
	if (n_idx == 2 )
	  {
	    idx_vector i = idx (0).index_vector ();

	    if (! error_state)
	      {
		idx_vector j = idx (1).index_vector ();

		if (! error_state)
		  retval = octave_value (matrix.index (i, j, resize_ok));
//		  class Octave_map;
//		  retval = Octave_map();
	RSBOI_DEBUG_NOTICE("UNFINISHED\n");
	      }
	  }
      }
      break;
    }

  return retval;
}
#endif

		virtual SparseMatrix sparse_matrix_value(bool = false)const
		{
			/* FIXME: UNFINISHED */
			struct rsb_coo_matrix_t coo;
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			rsb_nnz_index_t nnz=this->nnz();
			RSBOI_DEBUG_NOTICE("");
			Array<octave_idx_type> IA( dim_vector(1,nnz) );
			Array<octave_idx_type> JA( dim_vector(1,nnz) );
			Array<RSBOI_T> VA( dim_vector(1,nnz) );
			coo.VA=(RSBOI_T*)VA.data(),coo.IA=(rsb_coo_index_t*)IA.data(),coo.JA=(rsb_coo_index_t*)JA.data();
			errval=rsb_get_coo(this->A,coo.VA,coo.IA,coo.JA,RSB_FLAG_C_INDICES_INTERFACE);
			coo.m=this->rows();
			coo.k=this->cols();
			return SparseMatrix(VA,IA,JA,coo.m,coo.k);
		}

#if RSBOI_WANT_SUBSREF
		octave_value subsref (const std::string &type, const std::list<octave_value_list>& idx)
		{
			/* FIXME : incomplete implementation */
			octave_value retval;
			int skip = 1;
			RSBOI_DEBUG_NOTICE("");

			switch (type[0])
			{
				case '(':
		//		RSBOI_DEBUG_NOTICE("");
				if (type.length () == 1)
				{
  					octave_idx_type n_idx = idx.front().length ();
					if (n_idx == 2 )
	  				{
	    					idx_vector i = idx.front() (0).index_vector ();
	    					if (! error_state)
	      					{
							idx_vector j = idx.front() (1).index_vector ();
							RSBOI_T rv;
					  		octave_idx_type ii=-1,jj=-1;
							rsb_err_t errval=RSB_ERR_NO_ERROR;
  							ii=i(0);
 					 		jj=j(0);
							RSBOI_DEBUG_NOTICE("get_elements (%d %d)\n",ii,jj);
       							errval=rsb_get_elements(this->A,&rv,&ii,&jj,1,RSBOI_NF);
							retval=rv;
							if (! error_state)
							  ;//retval = octave_value (matrix.index (i, j, resize_ok));
//		  class Octave_map;
//		  retval = Octave_map();
//	RSBOI_DEBUG_NOTICE("UNFINISHED: set %d %d <- %lg\n",ii,jj,rhs.double_value());
//	RSBOI_DEBUG_NOTICE("UNFINISHED: set %d %d <- %lg\n",ii,jj,-99.99);
	      					}
	  				}
				}
				break;
				case '.':
					//retval = dotref (idx.front ())(0);
					RSBOI_DEBUG_NOTICE("UNFINISHED\n");
					break;

				case '{':
					error ("%s cannot be indexed with %c", type_name().c_str(), type[0]);
					//RSBOI_DEBUG_NOTICE("UNFINISHED\n");
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
			/* FIXME : missing implementation */
			octave_value_list retval;

			std::string nm = idx(0).string_value ();
			RSBOI_DEBUG_NOTICE("");

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

		// Need to define as map so tha "a.type = ..." can work
		bool is_map (void) const { return true; }
		bool is_sparse_type (void) const { RSBOI_DEBUG_NOTICE("");return true; }
		bool is_real_type (void) const { RSBOI_DEBUG_NOTICE("");return true; }
		bool is_bool_type (void) const { return false; }
		bool is_integer_type (void) const { return false; }
		bool is_square (void) const { return this->rows()==this->cols(); }
		bool is_empty (void) const { return false; }
//		int is_struct (void) const { return false; }

		octave_value subsasgn (const std::string& type, const std::list<octave_value_list>& idx, const octave_value& rhs)
		{
			/* FIXME:
			 * to complete, consult liboctave/idx-vector.h at line 500: index()
			 * and src/ov-base.cc at numeric_assign
			 * */
			octave_value retval;
#if 0
			error ("assignment is still unsupported on 'sparse_rsb' matrices");
			goto skipimpl;
#else
			RSBOI_DEBUG_NOTICE("");
#endif
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
									rsb_err_t errval=RSB_ERR_NO_ERROR;
									idx_vector j = idx.front() (1).index_vector ();
									octave_idx_type ii=-1,jj=-1;
									RSBOI_T rv=rhs.double_value();
									ii=i(0); jj=j(0);
									RSBOI_DEBUG_NOTICE("FIXME: UNFINISHED\n");
									RSBOI_DEBUG_NOTICE("update elements (%d %d)\n",ii,jj);
									errval=rsb_update_elements(this->A,&rv,&ii,&jj,1,RSBOI_NF);
									RSBOI_PERROR(errval);
									/* FIXME: I am unsure, here */
									//retval=rhs.double_value(); // this does not match octavej
									//retval=octave_value(this); 
									retval=octave_value(this->clone()); // matches but .. heavy ?!
									if (! error_state)
										;//retval = octave_value (matrix.index (i, j, resize_ok));
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
			/* FIXME : missing implementation */
			octave_base_value *retval = 0;
			RSBOI_DEBUG_NOTICE("");
			return retval;
		}

		//type_conv_fcn numeric_conversion_function (void) const
		type_conv_info numeric_conversion_function (void) const
		{
			RSBOI_DEBUG_NOTICE("");
			return default_numeric_conversion_function;
		}

	#if 0
		bool isupper (void) const { return false; /**/ }
		bool islower (void) const { return false; /**/ }

		void assign (const octave_value_list& idx, const Matrix& rhs)
		{
			/* FIXME : missing implementation */
			std::cerr << "octave_sparse_matrix::assign(idx, rhs);\n";
			//octave_sparse_matrix::assign(idx, rhs);
		}

		octave_sparse_rsb_matrix transpose (void) const
		{
			/* FIXME : missing implementation */
			return octave_sparse_rsb_matrix();
		}
	#endif

		void print (std::ostream& os, bool pr_as_read_syntax = false) const
		{
			/* FIXME : missing implementation */
			//      octave_sparse_matrix::print (os, pr_as_read_syntax);
			//os <<  " RSB Sparse *only in multiplication*";
#if 0
			RSBOI_DEBUG_NOTICE("");
			RSBOI_PRINTF("Recursive Sparse Blocks:\n");
			RSBOI_MP(this->A);
			RSBOI_PRINTF("\n");
			rsb_print_matrix_t(this->A);
#else
			/* FIXME: missing error checking */
			struct rsb_coo_matrix_t coo;
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			rsb_nnz_index_t nnz=this->nnz(),nzi;
			RSBOI_DEBUG_NOTICE("");
			Array<octave_idx_type> IA( dim_vector(1,nnz) );
			Array<octave_idx_type> JA( dim_vector(1,nnz) );
			Array<RSBOI_T> VA( dim_vector(1,nnz) );
			coo.VA=(RSBOI_T*)VA.data(),coo.IA=(rsb_coo_index_t*)IA.data(),coo.JA=(rsb_coo_index_t*)JA.data();
			if(coo.VA==NULL)
				nnz=0;
			else
				errval=rsb_get_coo(this->A,coo.VA,coo.IA,coo.JA,RSB_FLAG_C_INDICES_INTERFACE);
			coo.m=this->rows();
			coo.k=this->cols();
			octave_stdout<<"Recursive Sparse Blocks  (rows = "<<coo.m<<", cols = "<<coo.k<<", nnz = "<<nnz<<" ["<<100.0*(((RSBOI_T)nnz)/((RSBOI_T)coo.m))/coo.k<<"%])\n";
			for(nzi=0;nzi<nnz;++nzi)
				octave_stdout<<"  ("<<1+IA(nzi)<<", "<<1+JA(nzi)<<") -> "<<((RSBOI_T*)coo.VA)[nzi]<<"\n";
#endif
			newline(os);
done:			RSBIO_NULL_STATEMENT_FOR_COMPILER_HAPPINESS
		}

	octave_value diag (octave_idx_type k) const
	{
		octave_value retval;
		RSBOI_DEBUG_NOTICE("");
		if(this->is_square())
		{
			rsb_err_t errval=RSB_ERR_NO_ERROR;
			//Array<rsb_coo_index_t> DA( dim_vector(1,this->rows()) );
			Matrix DA(this->rows(),1);
			RSBOI_DEBUG_NOTICE("");
			errval=rsb_getdiag (this->A,(RSBOI_T*)DA.data());/*FIXME*/
			retval=DA;
			// FIXME: missing error handling
		}
		else
		{
			error("matrix is not square");
		}
//		if (k == 0 && matrix->ndims () == 2 && (matrix.rows () == 1 || matrix.columns () == 1))
//			;//retval = DiagMatrix (DiagArray2<double> (matrix));
//		else
			;//retval = octave_base_matrix<NDArray>::diag (k);
		return retval;
	}

	rsb_err_t rsboi_scale_inv(RSBOI_T alpha)
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		errval=rsb_elemental_scale_inv(this->A,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

	octave_value rsboi_get_scaled_copy_inv(const RSBOI_T alpha)const
	{
		octave_sparse_rsb_matrix * m = new octave_sparse_rsb_matrix(*this);
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		if(!m)return m;
		errval=m->rsboi_scale_inv(alpha);
		RSBOI_PERROR(errval);
		return m;
	}

	rsb_err_t rsboi_scale(RSBOI_T alpha)
	{
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		errval=rsb_elemental_scale(this->A,&alpha);
		RSBOI_PERROR(errval);
		return errval;
	}

	octave_value rsboi_get_scaled_copy(const RSBOI_T alpha)const
	{
		octave_sparse_rsb_matrix * m = new octave_sparse_rsb_matrix(*this);
		rsb_err_t errval=RSB_ERR_NO_ERROR;
		if(!m)return m;
		errval=m->rsboi_scale(alpha);
		RSBOI_PERROR(errval);
		return m;
	}

	private:

		DECLARE_OCTAVE_ALLOCATOR
			DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};/* end of class octave_sparse_rsb_matrix definition  */

#if 0
octave_value_list find_nonzero_elem_idx (const class octave_sparse_rsb_matrix & nda, int nargout, octave_idx_type n_to_find, int direction)
{
	// useless
	octave_value retval;
	RSBOI_DEBUG_NOTICE("");
	return retval;
}
#endif

static octave_base_value * default_numeric_conversion_function (const octave_base_value& a)
{
	/* FIXME : missing implementation */
	RSBOI_DEBUG_NOTICE("");
	CAST_CONV_ARG (const octave_sparse_rsb_matrix&);
	rsb_err_t errval=RSB_ERR_NO_ERROR;

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
	return new octave_sparse_matrix (v.sparse_matrix_value());
	#endif
}

DEFINE_OCTAVE_ALLOCATOR (octave_sparse_rsb_matrix);
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA (octave_sparse_rsb_matrix,
RSB_OI_TYPEINFO_STRING,
RSB_OI_TYPEINFO_TYPE);

DEFCONV (octave_triangular_conv, octave_sparse_rsb_matrix, matrix)
{
	/* FIXME : missing implementation */
	RSBOI_DEBUG_NOTICE("");
	CAST_CONV_ARG (const octave_sparse_rsb_matrix &);
	return new octave_sparse_matrix (v.matrix_value ());
}

#if 0
DEFCONV (octave_sparse_rsb_to_octave_sparse_conv, sparse_rsb_matrix, sparse_matrix)
{
	/* FIXME : missing implementation */
	RSBOI_DEBUG_NOTICE("");
	CAST_CONV_ARG (const octave_sparse_rsb_matrix &);
	return new octave_sparse_matrix (v.matrix_value ());
}
#endif

DEFUNOP (uplus, sparse_rsb_matrix)
{
	/* FIXME : missing implementation */
	RSBOI_DEBUG_NOTICE("");
	CAST_UNOP_ARG (const octave_sparse_rsb_matrix&);
	//std::cerr << "here\n";
	return new octave_sparse_rsb_matrix (v);
}

#if 0
DEFUNOP (op_incr, sparse_rsb_matrix)
{
	/* FIXME : missing implementation */
	RSBOI_DEBUG_NOTICE("");
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
	/* FIXME : missing implementation */
	RSBOI_DEBUG_NOTICE("");
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
	/* FIXME : missing implementation */
	RSBOI_DEBUG_NOTICE("");
	CAST_UNOP_ARG (const octave_sparse_rsb_matrix&);
	octave_sparse_rsb_matrix * m = new octave_sparse_rsb_matrix(v);
	if(!m)return m;
	rsb_negation(m->A);
	return m;
}

DEFUNOP (transpose, sparse_rsb_matrix)
{
	/* transpose(this->A) */
	RSBOI_DEBUG_NOTICE("");
	CAST_UNOP_ARG (const octave_sparse_rsb_matrix&);
	octave_sparse_rsb_matrix * m = new octave_sparse_rsb_matrix(v);
	/* TODO: here, the best solution would be to use some get_transposed() function */
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	if(!m)return m;
	errval=rsb_transpose(&m->A);
	RSBOI_PERROR(errval);
	return m;
	//  return new octave_sparse_rsb_matrix (v.transpose ());
	/* FIXME : work here */
	//return new octave_sparse_rsb_matrix ();
}

DEFUNOP (htranspose, sparse_rsb_matrix)
{
	/* transpose(this->A) */
	RSBOI_DEBUG_NOTICE("");
	CAST_UNOP_ARG (const octave_sparse_rsb_matrix&);
	octave_sparse_rsb_matrix * m = new octave_sparse_rsb_matrix(v);
	/* TODO: here, the best solution would be to use some get_transposed() function */
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	if(!m)return m;
	errval=rsb_htranspose(&m->A);
	RSBOI_PERROR(errval);
	return m;
	//  return new octave_sparse_rsb_matrix (v.transpose ());
	/* FIXME : work here */
	//return new octave_sparse_rsb_matrix ();
}

DEFBINOP(ldiv, sparse_rsb_matrix, matrix)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_matrix&);
	/*  const Matrix m = v1.matrix_value ();
	  const Matrix b = v2.matrix_value ();
	  octave_idx_type nr = m.rows ();
	  octave_idx_type nc = m.cols ();
	  octave_idx_type b_nc = b.cols ();
	*/
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	//octave_matrix v3=*(octave_matrix*)v2.clone();
	//Matrix retval=Matrix( v2.matrix_value().dims() );
	Matrix retval=Matrix( v2.matrix_value().dims() ,RSBOI_ZERO);
	const Matrix b = v2.matrix_value ();
	octave_idx_type b_nc = b.cols ();
	octave_idx_type b_nr = b.rows ();
	octave_idx_type ldb=b_nr;
	octave_idx_type ldc=v1.rows();
	octave_idx_type nrhs=b_nc;
	octave_idx_type nels=retval.rows()*retval.cols();
	for(octave_idx_type i=0;i<nels;++i)
		retval(i)=v2.matrix_value()(i);
	//Matrix retval=Matrix( v3.matrix_value());
	//retval=(v2.matrix_value());
	//Matrix retval=Matrix(  1,2);
	//octave_stdout << v2.matrix_value() << retval << "\n";
	RSBOI_DEBUG_NOTICE("");
	//errval=rsb_spsv(RSB_TRANSPOSITION_N,&rsboi_one,v1.A,(const RSBOI_T*)retval.data(),RSBOI_OV_STRIDE,(RSBOI_T*)retval.data(),RSBOI_OV_STRIDE);
	errval=rsb_spsm(RSB_TRANSPOSITION_N,&rsboi_one,v1.A,nrhs,RSB_OI_DMTXORDER,&rsboi_zero,(const RSBOI_T*)retval.data(),ldb,(RSBOI_T*)retval.data(),ldc);
	if(RSBOI_SOME_ERROR(errval))
	{
		if(errval==RSB_ERR_INVALID_NUMERICAL_DATA)
			RSBOI_PERROR(errval);// FIXME: need a specific error message here
		else
			RSBOI_PERROR(errval);// FIXME: generic case, here
		for(octave_idx_type i=0;i<nels;++i)
			retval(i)=octave_NaN;
	}
	//octave_stdout << v3.matrix_value() << retval << "\n";
	return retval;
}

DEFBINOP(el_div, sparse_rsb_matrix, matrix)
{
	Matrix retval;
	RSBOI_DEBUG_NOTICE("");
	/* FIXME : missing implementation */
	return retval;
}

DEFBINOP(el_ldiv, sparse_rsb_matrix, matrix)
{
	Matrix retval;
	RSBOI_DEBUG_NOTICE("");
	/* FIXME : missing implementation */
	return retval;
}

DEFBINOP(div, sparse_rsb_matrix, matrix)
{
	Matrix retval;
	RSBOI_DEBUG_NOTICE("");
	/* FIXME : missing implementation */
	return retval;
}

DEFBINOP(rsb_s_div, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE("");
	return v1.rsboi_get_scaled_copy_inv(v2.scalar_value());
}

DEFBINOP(rsb_s_mul, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE("");
	return v1.rsboi_get_scaled_copy(v2.scalar_value());
}

#if 0
DEFBINOP(rsb_s_pow, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE("");
	return v1.rsboi_get_power_copy(v2.scalar_value());
}
#endif

DEFASSIGNOP (assign, sparse_rsb_matrix, sparse_rsb_matrix)
{
	/* FIXME : I dunno how to trigger this! */
	CAST_BINOP_ARGS (octave_sparse_rsb_matrix &, const octave_sparse_rsb_matrix&);
	RSBOI_DEBUG_NOTICE("");
	rsb_assign(v1.A, v2.A);
	return octave_value();
}

DEFASSIGNOP (assignm, sparse_rsb_matrix, matrix)
{
	CAST_BINOP_ARGS (octave_sparse_rsb_matrix &, const octave_matrix&);
	RSBOI_DEBUG_NOTICE("");
	RSBOI_DESTROY(v1.A);
	octave_value retval;
	//v1.assign(idx, v2.matrix_value());
	v1=(idx, v2.matrix_value());
	//retval=v1;
	retval=v2.matrix_value();
	/* FIXME : missing implementation */
	return retval;
}

#if 0
DEFASSIGNOP(rsb_op_mul_eq_s, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (octave_sparse_rsb_matrix &, const octave_scalar&);
	octave_value retval;
	RSBOI_DEBUG_NOTICE("");
	RSBOI_PERROR(v1.rsboi_scale(v2.scalar_value()));
	retval=v1.matrix_value();
	return retval;
}

DEFASSIGNOP(rsb_op_div_eq_s, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (octave_sparse_rsb_matrix &, const octave_scalar&);
	octave_value retval;
	RSBOI_DEBUG_NOTICE("");
	RSBOI_PERROR(v1.rsboi_scale_inv(v2.scalar_value()));
	retval=v1.matrix_value();
	return retval;
}
#endif

DEFBINOP(rsb_el_mul_s, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE("");
	return v1.rsboi_get_scaled_copy(v2.scalar_value());
}

DEFBINOP(rsb_el_div_s, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE("");
	return v1.rsboi_get_scaled_copy_inv(v2.scalar_value());
}

DEFBINOP(el_pow, sparse_rsb_matrix, scalar)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix &, const octave_scalar&);
	RSBOI_DEBUG_NOTICE("");
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
	RSBOI_DEBUG_NOTICE("");
	v1.assign(idx, v2.matrix_value());
	/* FIXME : missing implementation */
	return octave_value();
}
#endif

DEFBINOP(op_sub, sparse_rsb_matrix, sparse_rsb_matrix)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_sparse_rsb_matrix&);
	octave_sparse_rsb_matrix*sm = new octave_sparse_rsb_matrix();
	octave_value retval = sm;
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	RSBOI_DEBUG_NOTICE("");
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
	RSBOI_DEBUG_NOTICE("");
	/* FIXME */
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
	RSBOI_DEBUG_NOTICE("");
	sm->A=rsb_matrix_mul(RSB_TRANSPOSITION_N,&rsboi_one,v1.A,RSB_TRANSPOSITION_N,&rsboi_one,v2.A,&errval);
	RSBOI_PERROR(errval);
	if(!sm->A)
		RSBOI_0_ERROR(RSBOI_0_ALLERRMSG);
	return retval;
}

DEFBINOP(op_mul, sparse_rsb_matrix, matrix)
{
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_matrix&);
	const Matrix b = v2.matrix_value ();
	//	const SparseMatrix sm = v1.sparse_matrix_value();
	octave_idx_type b_nc = b.cols ();
	octave_idx_type b_nr = b.rows ();
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	octave_idx_type ldb=b_nr;
	octave_idx_type ldc=v1.rows();
	octave_idx_type nrhs=b_nc;
	//Matrix retval( b.dims(),1,RSBOI_ZERO);
	Matrix retval(ldc,nrhs,RSBOI_ZERO);
	RSBOI_DEBUG_NOTICE("");
	if(v1.columns()!=b_nr) { error("matrices dimensions do not match!\n"); return Matrix(); }

	//errval=rsb_spmv(RSB_TRANSPOSITION_N,&rsboi_one,v1.A,(RSBOI_T*)b.data(),RSBOI_OV_STRIDE,&rsboi_one,(RSBOI_T*)retval.data(),RSBOI_OV_STRIDE);
	errval=rsb_spmm(RSB_TRANSPOSITION_N,&rsboi_one,v1.A,nrhs,RSB_OI_DMTXORDER,(RSBOI_T*)b.data(),ldb,&rsboi_zero,(RSBOI_T*)retval.data(),ldc);
	RSBOI_PERROR(errval);
	return retval;
}

DEFBINOP(op_trans_mul, sparse_rsb_matrix, matrix)
{
	// ".'*"  operator
	CAST_BINOP_ARGS (const octave_sparse_rsb_matrix&, const octave_matrix&);
	const Matrix b = v2.matrix_value ();
	octave_idx_type b_nc = b.cols ();
	octave_idx_type b_nr = b.rows ();
	rsb_err_t errval=RSB_ERR_NO_ERROR;
	octave_idx_type ldb=b_nr;
	octave_idx_type ldc=v1.columns();
	octave_idx_type nrhs=b_nc;
	Matrix retval(ldc,nrhs,RSBOI_ZERO);
	RSBOI_DEBUG_NOTICE("");
	if(v1.rows()!=b_nr) { error("matrices dimensions do not match!\n"); return Matrix(); }
	//octave_stdout << "have: ldc=" <<ldc << " bc=" << b_nc<< " nrhs=" << nrhs << " retval="<< retval<< "\n";

	//errval=rsb_spmv(RSB_TRANSPOSITION_T,&rsboi_one,v1.A,(RSBOI_T*)b.data(),RSBOI_OV_STRIDE,&rsboi_one,(RSBOI_T*)retval.data(),RSBOI_OV_STRIDE);
	errval=rsb_spmm(RSB_TRANSPOSITION_T,&rsboi_one,v1.A,nrhs,RSB_OI_DMTXORDER,(RSBOI_T*)b.data(),ldb,&rsboi_zero,(RSBOI_T*)retval.data(),ldc);
	RSBOI_PERROR(errval);
	return retval;
}

static void install_sparsersb_ops (void)
{
	//RSBOI_DEBUG_NOTICE("");
	/* FIXME : missing implementation */
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
	INSTALL_BINOP (op_mul, octave_sparse_rsb_matrix, octave_sparse_rsb_matrix, op_spmul);
	INSTALL_BINOP (op_trans_mul, octave_sparse_rsb_matrix, octave_matrix, op_trans_mul);
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
	RSBOI_DEBUG_NOTICE("");
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

	if (!sparse_rsb_loaded)
	{
		octave_sparse_rsb_matrix::register_type ();
		install_sparsersb_ops ();
		sparse_rsb_loaded=true;
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
Create a sparse RSB matrix by loading the Matrix Market matrix file from string @var{filename}.\n"\
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
s = "RSBOI_FNS" (i, j, s, m, n, \"sum\")\n\
s = "RSBOI_FNS" (i, j, s, \"summation\")\n\
s = "RSBOI_FNS" (i, j, s, \"sum\")\n\
@end group\n\
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
"@seealso{full}\n\
@end deftypefn")
{
	/* FIXME: TODO: implement all of the constructors */
	int nargin = args.length ();
	octave_value_list retval;
	octave_sparse_rsb_matrix*matrix=NULL;
	bool ic=nargin>0?(args(0).is_complex_type()):false;

	RSBOI_DEBUG_NOTICE("in sparsersb()\n");

	install_sparse_rsb();
	if (nargin == 1 || nargin == 2)
	{
		rsb_type_t typecode=RSBOI_TYPECODE;
		if (nargin >= 2)
#ifdef RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX
			typecode=RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX;
#else
			RSBOI_ERROR("compiled without complex type support!\n");
#endif

		if(args(0).is_sparse_type())
		{

			if(args(0).type_name()==RSB_OI_TYPEINFO_STRING)
			{
				retval.append(matrix=(octave_sparse_rsb_matrix*)(args(0).get_rep()).clone());
			}
			else
			{
				const SparseMatrix m = args(0).sparse_matrix_value();
				if (error_state) goto err;
				retval.append(matrix=new octave_sparse_rsb_matrix(m,typecode));
			}
		}
		else
		if(args(0).is_string())
		{
			const std::string m = args(0).string_value();
			if (error_state) goto err;
			retval.append(matrix=new octave_sparse_rsb_matrix(m));
		}
		else
		{
			Matrix m = args(0).matrix_value();
			if (error_state) goto err;
			retval.append(matrix=new octave_sparse_rsb_matrix(m));
		}
	}
	else
	if (nargin == 3)
	{
		retval.append(matrix=new octave_sparse_rsb_matrix( args(0).matrix_value(), args(1).matrix_value(), args(2).matrix_value() ));
	}
	else
	if (nargin == 5)
	{
		retval.append(matrix=new octave_sparse_rsb_matrix( args(0).matrix_value(), args(1).matrix_value(), args(2).matrix_value() ));
	}
	else
	if (nargin == 4  && args(3).is_string())
	{
		rsb_flags_t eflags;
		std::string vv= args(3).string_value();
		if (error_state) goto ret;
		RSBOI_DEBUG_NOTICE("");
		if ( vv == "summation" || vv == "sum" )
			eflags=RSB_FLAG_DUPLICATES_SUM;
		else
		if ( vv == "unique" )
			eflags=RSB_FLAG_DUPLICATES_KEEP_LAST;
		else
			goto errp;
		retval.append(matrix=new octave_sparse_rsb_matrix( args(0).matrix_value(), args(1).matrix_value(), args(2).matrix_value(), eflags));
	}
	else
	if (nargin == 6  && args(5).is_string())
	{
		rsb_flags_t eflags;
		std::string vv= args(5).string_value();
		if (error_state) goto ret;
		RSBOI_DEBUG_NOTICE("");
		if ( vv == "summation" || vv == "sum" )
			eflags=RSB_FLAG_DUPLICATES_SUM;
		else
		if ( vv == "unique" )
			eflags=RSB_FLAG_DUPLICATES_KEEP_LAST;
		else
			goto errp;
		retval.append(matrix=new octave_sparse_rsb_matrix( args(0).matrix_value(), args(1).matrix_value(), args(2).matrix_value(), eflags));
	}
	else
		goto errp;
	if(!matrix)
	{
		// TODO: error handling
		RSBOI_DEBUG_NOTICE("fatal error! matrix NOT built!\n");
	}
#if RSBOI_WANT_HEAVY_DEBUG
	if(!rsb_is_correctly_built_rcsr_matrix(matrix->A)) // non-declared function
	{
		// TODO: need error handling
		RSBOI_DEBUG_NOTICE("matrix NOT correctly built!\n");
	}
#endif
	goto err;
	errp:
	print_usage ();
err:
ret:
	return retval;
}
