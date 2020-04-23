// modified from cupy-generated reduction kernel code
#include <cupy/complex.cuh>
#include <cupy/carray.cuh>
#include <cupy/atomics.cuh>

typedef double T;

#define REDUCE(a, b) (a + b)
#define POST_MAP(abcd, ab, ac, ad, bc, bd, cd) (out = abcd/length-ab/length*cd/length-ac/length*bd/length-ad/length*bc/length)
#define _REDUCE(_offset) if (_tid < _offset) {   _type_reduce _a = _sdata[_tid], _b = _sdata[(_tid + _offset)];   _sdata[_tid] = REDUCE(_a, _b); }

typedef T _type_reduce;
extern "C" __global__ void reduce_kernel(const CArray<double, 1> _raw_A, const CArray<double, 1> _raw_B, const CArray<double, 1> _raw_C, const CArray<double, 1> _raw_D, const double length, CArray<double, 1> _raw_out, CIndexer<1> _in_ind, CIndexer<1> _out_ind, const int _block_stride) {
  __shared__ char _sdata_raw[512 * sizeof(_type_reduce)];
  _type_reduce *_sdata= reinterpret_cast<_type_reduce*>(_sdata_raw);
  unsigned int _tid = threadIdx.x;

  int _J_offset = _tid >> __popc(_block_stride - 1);  // _tid / _block_stride
  ptrdiff_t _j_offset = (ptrdiff_t)_J_offset * _out_ind.size();
  int _J_stride = 512 >> __popc(_block_stride - 1);
  ptrdiff_t _j_stride = (ptrdiff_t)_J_stride * _out_ind.size();

  for (ptrdiff_t _i_base = (ptrdiff_t)blockIdx.x * _block_stride;
       _i_base < _out_ind.size();
       _i_base += (ptrdiff_t)gridDim.x * _block_stride) {
    _type_reduce _s_abcd = _type_reduce(0);
    _type_reduce _s_ab= _type_reduce(0);
    _type_reduce _s_ac = _type_reduce(0);
    _type_reduce _s_ad = _type_reduce(0);
    _type_reduce _s_bc = _type_reduce(0);
    _type_reduce _s_bd = _type_reduce(0);
    _type_reduce _s_cd = _type_reduce(0);
    ptrdiff_t _i =
        _i_base + (_tid & (_block_stride - 1));  // _tid % _block_stride
    int _J = _J_offset;
    for (ptrdiff_t _j = _i + _j_offset; _j < _in_ind.size();
         _j += _j_stride, _J += _J_stride) {
      _in_ind.set(_j);
      const T A = _raw_A[_in_ind.get()];
      const T B = _raw_B[_in_ind.get()];
      const T C = _raw_C[_in_ind.get()];
      const T D = _raw_D[_in_ind.get()];
      _type_reduce _a_abcd = static_cast<_type_reduce>(A*B*C*D);
      _type_reduce _a_ab = static_cast<_type_reduce>(A*B);
      _type_reduce _a_ac = static_cast<_type_reduce>(A*C);
      _type_reduce _a_ad = static_cast<_type_reduce>(A*D);
      _type_reduce _a_bc = static_cast<_type_reduce>(B*C);
      _type_reduce _a_bd = static_cast<_type_reduce>(B*D);
      _type_reduce _a_cd = static_cast<_type_reduce>(C*D);
      _s_abcd = REDUCE(_s_abcd, _a_abcd);
      _s_ab= REDUCE(_s_ab, _a_ab);
      _s_ac = REDUCE(_s_ac, _a_ac);
      _s_ad = REDUCE(_s_ad, _a_ad);
      _s_bc = REDUCE(_s_bc, _a_bc);
      _s_bd = REDUCE(_s_bd, _a_bd);
      _s_cd = REDUCE(_s_cd, _a_cd);
    }
    _sdata_abcd[_tid] = _s_abcd;
    _sdata_ab[_tid] = _s_ab;
    _sdata_ac[_tid] = _s_ac;
    _sdata_ad[_tid] = _s_ad;
    _sdata_bc[_tid] = _s_bc;
    _sdata_bd[_tid] = _s_bd;
    _sdata_cd[_tid] = _s_cd;
    __syncthreads();
    for (unsigned int _block = 512 / 2;
         _block >= _block_stride; _block >>= 1) {
      if (_tid < _block) {
        _REDUCE(_block);
      }
      __syncthreads();
    }
    if (_tid < _block_stride) {
      _s_abcd = _sdata_abcd[_tid];
      _s_ab = _sdata_ab[_tid];
      _s_ac = _sdata_ac[_tid];
      _s_ad = _sdata_ad[_tid];
      _s_bc = _sdata_bc[_tid];
      _s_bd = _sdata_bd[_tid];
      _s_cd = _sdata_cd[_tid];
    }
    if (_tid < _block_stride && _i < _out_ind.size()) {
      _out_ind.set(static_cast<ptrdiff_t>(_i));
      T &out = _raw_out[_out_ind.get()];
      POST_MAP(_s_abcd, _s_ab, _s_ac, _s_ad, _s_bc, _s_bd, _s_cd);
    }
  }
}