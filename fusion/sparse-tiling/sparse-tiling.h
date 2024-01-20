//
// Created by kazem on 1/19/24.
//

#ifndef SPARSE_FUSION_SPARSE_TILING_H
#define SPARSE_FUSION_SPARSE_TILING_H

#include "SWTensorBench.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Utils.h"
#include "sparse-fusion/MultiDimensionalSet.h"
#include <omp.h>

namespace sym_lib {


CSC* merge_graph_two(int ngraphs, int n, int **Gps, int **Gis) {
  const int *Gp, *Gi;
  int nnz = 0;
  for(int i = 0; i < ngraphs; i++)
    nnz += Gps[i][n];
  nnz += (ngraphs-1) * n;
  CSC *merged_graph = new CSC(n*ngraphs,n*ngraphs,nnz, true);
  /** allocate new graph space **/
  int *nGp = merged_graph->p;
  int *nGi = merged_graph->i;
  nGp[0] = 0;
  for (int k = 1; k < n+1; ++k) {
    nGp[k] = nGp[k-1];
    for (int i = 0; i < ngraphs; ++i) {
      nGp[k] += (Gps[i][k] - Gps[i][k-1]);
    }
    //nGp[k] ++; //= (ngraphs - 1);
    auto t_idx = nGp[k-1];
    for (int i = 0; i < ngraphs; ++i) {
      for (int j = Gps[i][k-1]; j < Gps[i][k]; ++j) {
        nGi[t_idx] = Gis[i][j] + i*n;
        t_idx++;
      }
      //nGi[t_idx] = (i+1) * n;//extra edge
      //t_idx++;
    }
    assert(nGp[k] == t_idx);
  }

  return merged_graph;
}


struct vertex{
  int id, kernel_id;
  vertex():id(0),kernel_id(0){}
  vertex(int i, int k):id(i),kernel_id(k){}
};

bool vertex_cmp(vertex a, vertex b){
  if(a.kernel_id < b.kernel_id)
    return true;
  if(a.kernel_id == b.kernel_id){
    return a.id < b.id;
  }
  if(a.kernel_id > b.kernel_id)
    return false;
}

inline void sparseTilingMvMv(CSC *A,
                         std::vector<std::vector<std::vector<vertex>>>& Schedule,
                         int SeedNo=4){
  int dim = A->n;
  int nGraphs = 2;
  auto *tmpA = new CSC(A->m,A->n,A->n);//DAG of A
  for (int i = 0; i < dim; ++i) {
    tmpA->p[i]=i;
    tmpA->i[i]=i;
  }
  tmpA->p[dim]=dim;
  int **aps = new int*[nGraphs];
  int **ais = new int*[nGraphs];
  aps[1]=tmpA->p; aps[0]=A->p;
  ais[1]=tmpA->i; ais[0]=A->i;
  CSC *combinedDag;
  combinedDag = merge_graph_two(nGraphs, dim, aps, ais);
  print_csc(1,"combined\n",combinedDag);
  //print_vec(" \nvec\n",0, dim+1, combined_DAG->p);
  delete []aps;
  delete []ais;
  auto *vK2Exist = new bool[dim]();
  auto *vK1Exist = new bool[dim]();
  std::vector<bool *> vertexExist;
  auto rowVec = new bool[combinedDag->m]();
  vertexExist.push_back(rowVec);
  std::vector<std::vector<vertex>> emptyRow1;
  Schedule.push_back(emptyRow1);
  int i = 0, curLevel=0;
  while(i < dim){
    curLevel=0;
    std::vector<vertex> tmp;
    for (int j = i; j < std::min(i + SeedNo,dim); ++j) {
      if(!vK1Exist[j]){
        tmp.emplace_back(j,0);//vertex of the first loop
        vK1Exist[j] = true;
      }
      // bfs
      while(curLevel < vertexExist.size()){
        if(vertexExist[curLevel][j]) // vertex exists, correctness issue
          curLevel++;
        else
          break;
      }
      for (int k = combinedDag->p[j]+1; k < combinedDag->p[j + 1]; ++k) {
        auto vtx = combinedDag->i[k]%dim;
        auto type = combinedDag->i[k] / dim;
        auto vvtx = combinedDag->i[k];
        // if one vertex of the second DAG exist in the current level
        // go to the next level
        int tmpCurLev=0; bool sw=false;
        while(tmpCurLev < vertexExist.size()){
          if(!vertexExist[tmpCurLev][vvtx]){ // vertex exists, correctness issue
            tmpCurLev++;
          }else{
            sw = true;
            tmpCurLev++;
            break;
          }
        }
        if(sw)
          curLevel = std::max(tmpCurLev, curLevel);
        if(type == 0 ){
          if(!vK1Exist[vtx]){
            tmp.emplace_back(vtx,0);
            vK1Exist[vtx] = true;
          }
        }
        if(!vK2Exist[vtx] && type == 1){// second kernel
          tmp.emplace_back(vtx,1);
          vK2Exist[vtx] = true;
        }
      }
    }
    // now copy tmp to the right level
    assert(curLevel <= vertexExist.size());
    if(curLevel == vertexExist.size()){// allocate a new row
      auto *tmpRow = new bool[combinedDag->m]();
      vertexExist.push_back(tmpRow);
      std::vector<std::vector<vertex>> emptyRow; Schedule.push_back(emptyRow);
    }
    std::sort(tmp.begin(), tmp.end(), vertex_cmp);
    if(!tmp.empty())
      Schedule[curLevel].push_back(tmp);
    for (int l = 0; l < tmp.size(); ++l) {
      vertexExist[curLevel][tmp[l].id + (tmp[l].kernel_id*dim)]=true;
    }
    i += SeedNo;
  }
  std::vector<vertex> tmp;
  for (int m = 0; m < dim; ++m) {
    if(!vK2Exist[m]){//isolated nodes will run at the first level
      tmp.emplace_back(m,1);
    }
  }
  if(!tmp.empty())
    Schedule[0].push_back(tmp);
  for (int l = 0; l < vertexExist.size(); ++l) {
    delete []vertexExist[l];
  }
  delete []vK1Exist;
  delete []vK2Exist;
  delete combinedDag;
  delete tmpA;
}

inline void convertScheduleToSet(std::vector<std::vector<std::vector<vertex>>> Sched,
                             int &Levels, int *SparPtr, int *WparPtr, int *Vert, int *TypeV){
  Levels = Sched.size();
  int wparCnt=1, sparCnt=1, vertxCnt=0;
  SparPtr[0] = 0; WparPtr[0]=0;
  Levels = Sched.size();
  for (int i = 0; i < Sched.size(); ++i) {
    for (int j = 0; j < Sched[i].size(); ++j) {
      for (int k = 0; k < Sched[i][j].size(); ++k) {
        Vert[vertxCnt] = Sched[i][j][k].id;
        TypeV[vertxCnt] = Sched[i][j][k].kernel_id;
        vertxCnt++;
      }
      WparPtr[wparCnt] = vertxCnt;
      wparCnt++;
    }
    SparPtr[sparCnt] = wparCnt-1;
    sparCnt++;
  }
}


inline void spmvCsrSpmvCsrSparseTiling(int n, const int *Ap,
                                      const int *Ai,
                                      const double *Ax, double *x,
                                      double *z,
                                      double *y,
                                      int LevelNo,
                                      const int *LevelPtr,
                                      const int *ParPtr,
                                      const int *Partition,
                                      const int *ParType,
                                      double *Tmp) {
  timing_measurement t1, t2;
  for (int i1 = 0; i1 < LevelNo; ++i1) {
#pragma omp parallel
    {
#pragma omp  for schedule(dynamic) nowait
      for (int j1 = LevelPtr[i1]; j1 < LevelPtr[i1 + 1]; ++j1) {
        //int tid = omp_get_thread_num();
        //double *tt = tmp + tid * n;
        //std::fill_n(tt, n, 0);
        for (int k1 = ParPtr[j1]; k1 < ParPtr[j1 + 1]; ++k1) {
          int i = Partition[k1];
          int t = ParType[k1];
          if (t == 0) {
            //    t2.start_timer();
            //spmv
            double tmp_val=0;
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
              tmp_val += Ax[j] * x[Ai[j]];// in tmp
            }
            z[i] = tmp_val; //tt[i];
          } else {
            //print_vec("TT \n",0,n,tt);
            //y[i] += tt[i]; // fully summed
            // spmv
            //std::cout<<" i: "<< i<<" z: "<<z[i];
            // std::cout<<i1<<" ,"<<j1<<" ,"<< i << ", " <<y[i] <<" + "<< z[i] <<" - " <<Lx[Lp[i ]]<<"\n";
            for (int k = Ap[i]; k < Ap[i + 1]; k++) {
              y[i] += Ax[k] * z[Ai[k]];
            }
          }
        }
      }
    }
  }
}





}

#endif // SPARSE_FUSION_SPARSE_TILING_H
