//
// Created by salehm32 on 27/01/24.
//

#ifndef SPARSE_FUSION_GCN_DEF_H
#define SPARSE_FUSION_GCN_DEF_H

#define NULLPNTR nullptr

#include <stdlib.h>

template<class T>
struct CSCType {
  size_t m; // rows
  size_t n; // columns
  size_t nnz; // nonzeros
  int stype;
  bool is_pattern;
  bool pre_alloc; //if memory is allocated somewhere other than const.
  int *p; // Column pointer array
  int *i; // Row index array
  T *x;

  CSCType(size_t M, size_t N, size_t NNZ) : m(M), n(N), nnz(NNZ) {
    is_pattern = false;
    pre_alloc = false;
    if (N > 0)
      p = new int[N + 1]();
    else
      p = NULLPNTR;
    if (NNZ > 0) {
      i = new int[NNZ]();
      x = new T[NNZ]();
    } else {
      i = NULLPNTR;
      x = NULLPNTR;
    }
    stype = 0;
  };

  CSCType(size_t M, size_t N, size_t NNZ, bool ip) :
                                                     m(M), n(N), nnz(NNZ), is_pattern(ip) {
    is_pattern = ip;
    pre_alloc = false;
    if (N > 0)
      p = new int[N + 1]();
    else
      p = NULLPNTR;
    if (NNZ > 0) {
      i = new int[NNZ]();
      if (!is_pattern)
        x = new T[NNZ]();
      else
        x = NULLPNTR;
    } else {
      i = NULLPNTR;
      x = NULLPNTR;
    }
    stype = 0;
  };

  CSCType(size_t M, size_t N, size_t NNZ, bool ip, int st) :
                                                             m(M), n(N), nnz(NNZ), is_pattern(ip) {
    is_pattern = ip;
    pre_alloc = false;
    if (N > 0)
      p = new int[N + 1]();
    else
      p = NULLPNTR;
    if (NNZ > 0) {
      i = new int[NNZ]();
      if (!is_pattern)
        x = new T[NNZ]();
      else
        x = NULLPNTR;
    } else {
      i = NULLPNTR;
      x = NULLPNTR;
    }
    stype = st;
  };

  CSCType(size_t M, size_t N, size_t NNZ, int *Ap, int *Ai, T *Ax) {
    is_pattern = false;
    pre_alloc = true;
    m = M;
    n = N;
    nnz = NNZ;
    p = Ap;
    i = Ai;
    x = Ax;
  }

  CSCType(size_t M, size_t N, size_t NNZ, int *Ap, int *Ai, int st) {
    is_pattern = true;
    pre_alloc = true;
    m = M;
    n = N;
    nnz = NNZ;
    p = Ap;
    i = Ai;
    x = NULLPNTR;
    stype = st;
  }

  ~CSCType() {
    if (!pre_alloc) {
      if (n > 0)
        delete[]p;
      if (nnz > 0) {
        delete[]i;
        if (!is_pattern)
          delete[]x;
      }
    }
  }

};

template<class T>
struct CSRType {
  size_t m; // rows
  size_t n; // columns
  size_t nnz; // nonzeros
  int stype;
  bool is_pattern;
  bool pre_alloc; //if memory is allocated somewhere other than const.
  int *p; // Row pointer array
  int *i; // Column index array
  T *x;

  CSRType(size_t M, size_t N, size_t NNZ) : m(M), n(N), nnz(NNZ) {
    is_pattern = false;
    pre_alloc = false;
    if (M > 0)
      p = new int[M + 1]();
    else
      p = NULLPNTR;
    if (NNZ > 0) {
      i = new int[NNZ]();
      x = new T[NNZ]();
    } else {
      i = NULLPNTR;
      x = NULLPNTR;
    }
    stype = 0;
  }

  CSRType(size_t M, size_t N, size_t NNZ, bool ip) :
                                                     m(M), n(N), nnz(NNZ), is_pattern(ip) {
    is_pattern = ip;
    pre_alloc = false;
    if (M > 0)
      p = new int[M + 1]();
    else
      p = NULLPNTR;
    if (NNZ > 0) {
      i = new int[NNZ]();
      if (!is_pattern)
        x = new T[NNZ]();
      else
        x = NULLPNTR;
    } else {
      i = NULLPNTR;
      x = NULLPNTR;
    }
    stype = 0;
  };

  CSRType(size_t M, size_t N, size_t NNZ, int *Ap, int *Ai, int st) {
    is_pattern = true;
    pre_alloc = true;
    m = M;
    n = N;
    nnz = NNZ;
    p = Ap;
    i = Ai;
    x = NULLPNTR;
    stype = st;
  }

  ~CSRType() {
    if (!pre_alloc) {
      if (m > 0)
        delete[]p;
      if (nnz > 0) {
        delete[]i;
        delete[]x;
      }
    }
  }

};



#endif // SPARSE_FUSION_GCN_DEF_H
