//
// Created by Kazem on 2023-02-18.
//
#include "sparse-fusion/Fusion_Utils.h"
#include "aggregation/def.h"
#include "aggregation/lbc.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "aggregation/test_utils.h"
#include <cstdlib>
#include <fstream>
#include <sstream>

#ifdef METIS
#include "aggregation/metis_interface.h"
#endif

namespace sym_lib{


 void parse_args(const int Argc, const char **Argv, ScheduleParameters *Sp,
                 TestParameters *Tp){
  if(Argc <2){
   Tp->_mode = "Random";
   Tp->_dim1 = Tp->_dim2 = 16;
   Tp->_matrix_name = "Random_"+ std::to_string(Tp->_dim1);
   Tp->_order_method=SYM_ORDERING::NONE; Tp->print_header=true;
   Sp->_num_threads=3;
   Sp->_num_w_partition = Sp->_num_threads; Sp->_lbc_agg=Sp->_lbc_initial_cut=4;
   return;
  }
  Sp->_num_threads = 6;
  Tp->_matrix_path = Argv[1];
  if(Argc >= 3)
   Sp->_num_threads = atoi(Argv[2]);
  int useLevelCoarsening = 1;
  if(Argc >= 4)
   useLevelCoarsening = atoi(Argv[3]);
  if(Argc >= 5)
   Sp->_lbc_agg = atoi(Argv[4]);
  Tp->_mode = "MTX";
  Tp->_matrix_name = Tp->_matrix_path.substr(Tp->_matrix_path.find_last_of("/\\") + 1);;
  if(Argc >= 6)
   Tp->print_header = atoi(Argv[5]);
  if(Argc >= 7)
   Tp->_b_cols = atoi(Argv[6]);
  if(Argc >= 8)
   Sp->IterPerPartition = atoi(Argv[7]);
  if(Argc >= 9)
   Sp->TileN = atoi(Argv[8]);
 }


 int get_reorderd_matrix(const CSC *L1_csc, std::vector<CSC*>& mat_vec){
  /// Re-ordering L matrix
#ifdef METIS
  int *perm=NULLPNTR;
  CSC *Lt_ordered_csc; CSC *L_ordered_csc;
  //We only reorder L since dependency matters more in l-solve.
  //perm = new int[n]();
  CSC *L1_csc_full = make_full((CSC*)L1_csc);
  metis_perm_general(L1_csc_full, perm);
  auto L1_csc_half = make_half(L1_csc_full->n, L1_csc_full->p, L1_csc_full->i,
                               L1_csc_full->x);
  Lt_ordered_csc = transpose_symmetric((CSC*)L1_csc_half, perm);
  L_ordered_csc = transpose_symmetric(Lt_ordered_csc, NULLPNTR);
  mat_vec.push_back(Lt_ordered_csc); mat_vec.push_back(L_ordered_csc);
  delete L1_csc_full;
  delete L1_csc_half;
  delete[]perm;
  return 0;
#endif
  return -1;
 }

 sym_lib::CSR* readSparseMatrix(const std::string &Path) {
  std::ifstream file;
  file.open(Path, std::ios_base::in);
  if (!file.is_open()) {
   std::cout << "File could not be found..." << std::endl;
   exit(1);
  }

//  if (!std::is_same_v<type, CSR>) {
//   throw std::runtime_error("Error: Matrix storage format not supported");
//  }

  std::string line;

  std::getline(file, line);
  std::replace(line.begin(), line.end(), ',', ' ');
  std::istringstream firstLine(line);

  int rows, cols, nnz;
  firstLine >> rows;
  firstLine >> cols;
  firstLine >> nnz;

  CSR *csr = new CSR(rows, cols, nnz);

  for (int i = 0; i < csr->m + 1; i++) {
   file >> csr->p[i];
  }

  // Go to next line
  char next;
  while (file.get(next)) { if (next == '\n') break; }

  // Read in col_indices
  for (int i = 0; i < csr->nnz; i++) { file >> csr->i[i]; }
  for (int i = 0; i < csr->nnz; i++) { csr->x[i] = 1.0f; }

  return csr;
 }

 CSC* get_matrix_from_parameter(const TestParameters *TP){
  CSC *aCSC=NULLPNTR;
  if (TP->_mode == "Random") {
   aCSC = random_square_sparse(TP->_dim1, TP->_density);
  } else {
   std::string fileExt = TP->_matrix_path.substr(
       TP->_matrix_path.find_last_of(".") + 1);
   if (fileExt == "smtx") {
      auto *tmpCsr = readSparseMatrix(TP->_matrix_path);
      aCSC = csr_to_csc(tmpCsr);
      delete tmpCsr;
   } else if (fileExt == "mtx") {
     std::ifstream fin(TP->_matrix_path);
     sym_lib::read_mtx_csc_real(fin, aCSC);
   } else{
    std::cout << "File extension is not supported" << std::endl;
    exit(1);
   }
  }
  return aCSC;
 }

 // starts from in_set in G1 and reaches to all unvisited vertices in G2
// G1 -> G2 , D is transpose of dependence works as a func from G2 to G1
// visited_g1 shows visited vertices of previous coarsened wf
 void forward_pairing(CSC *G2, CSC *D, const std::vector<int>& InSet, std::vector<int>& OutSet,
                      const std::vector<bool>& VisitedG1, const std::vector<bool>& VisitedG2){
  std::vector<bool> tmpG1(G2->m, false);
  for (int id : InSet) {
   tmpG1[id] = true; // each vertex in in_set is visited locally
  }
  for (int id : InSet) {
   //visited_g2[id] = true; // each node is visited itself
   //tmpG1[id] = true; // each vertex in in_set is visited locally
   bool selfDisjoint = true;
   // check all dependent G2 vertices are there
   for (int j = G2->p[id]; j < G2->p[id+1]; ++j) {
    auto reachedIter = G2->i[j];
    if(!VisitedG2[reachedIter] && reachedIter != id){
     selfDisjoint = false;
     break;
    }
   }
   if(selfDisjoint){
    // check all dependent G1 vertices are there
    for (int j = D->p[id]; j < D->p[id+1]; ++j) {
     auto reachedIter = D->i[j]; // an iteration of G1
     //if(!VisitedG1[reachedIter]){
     if(!tmpG1[reachedIter]){
      selfDisjoint = false;
      break;
     }
    }
    if(selfDisjoint)
     OutSet.push_back(id); // id can be added now
   }
  }
 }


 /// Calculates the summation of a vector
 /// \tparam type
 /// \param n
 /// \param vec
 /// \return
 template<class type> type sumVector(int N, type *Vec){
  double sum = 0;
  for (int i = 0; i < N; ++i) sum += Vec[i];
  return sum;
 }

 /// Partitions the set into n_parts based on the weight of each element
 void partitionByWeight(int N,const int *Set, const double *Weight,
                        int NParts,
                        double *TargetWeight,
                        std::vector<int> &Indices){
  double *evenWeight;
  if(TargetWeight)
   evenWeight = TargetWeight;
  else{
   evenWeight = new double[NParts];
   double evenW = std::ceil(sumVector(N, Weight) / NParts);
   std::fill_n(evenWeight, NParts, evenW);
  }
  Indices.resize(NParts+1);
  int j = 0;
  for (int i = 0; i < NParts; ++i) {
   double cWgt = 0;
   while (cWgt < evenWeight[i] && j < N){
    int cN = Set[j];
    cWgt += Weight[cN];
    j++;
   }
   Indices[i+1] = j;
  }
  if(!TargetWeight)
   delete []evenWeight;
 }


 void partitionByBfs(int N, const CSC *Df, const double *Weight,
                     double *TargetWeight,
                     int PartNo,
                     std::vector<int>  &FinalPartPtr, std::vector<int> &FinalNodePtr
                       ){
  double *evenWeight;
  //FinalLevelNo = 1;
  //FinaLevelPtr = new int[FinalLevelNo+1]();
  //FinaLevelPtr[FinalLevelNo] = PartNo;
  //FinalPartPtr = new int[PartNo+1]();
  //FinalNodePtr = new int[N]();
  if(TargetWeight)
   evenWeight = TargetWeight;
  else{
   evenWeight = new double[PartNo];
   double evenW = std::ceil(sumVector(N, Weight) / PartNo);
   std::fill_n(evenWeight, PartNo, evenW);
  }
  auto *isVisited = new bool[N](); int visitedNodes=0;
  int nxtStart = 0; std::vector<std::vector<int>> setTemp;
  setTemp.resize(PartNo);
  std::vector<int> stack;
  for (int i = 0; i < PartNo && visitedNodes<N; ++i) {
   assert(stack.size() < N);
   double cWgt = 0;
   while (cWgt < evenWeight[i] && visitedNodes < N){
    if(stack.empty()){
     nxtStart=-1;
     for (int k = 0; k < N; ++k) {
      if(!isVisited[k]){
        nxtStart = k;
        break;
      }
     }
     if(nxtStart < 0) break;
     stack.push_back(nxtStart);
     isVisited[nxtStart] = true;
    }
    // do bfs per node nxt_start
    while(!stack.empty()){
     int cn = stack[0]; stack.erase(stack.begin());
     cWgt += Weight[cn]; setTemp[i].push_back(cn);visitedNodes++;
     for (int k = Df->p[cn]; k < Df->p[cn + 1]; ++k) {
      auto cnn = Df->i[k];
      if(!isVisited[cnn]){
        stack.push_back(cnn);
        isVisited[cnn] = true;
        for (int j = Df->p[cnn]; j < Df->p[cnn + 1]; ++j) {
          auto newCnn = Df->i[j];
          if(!isVisited[newCnn]){
           stack.push_back(newCnn);
           isVisited[newCnn] = true;
          }
        }
      }
     }
     if(cWgt >= evenWeight[i])
      break;
    }
   }
  }
  // if anything is remained add it to last part
  for (int m = 0; m < stack.size(); ++m) {
   setTemp[PartNo-1].push_back(stack[m]);
  }
  for (int k = 0; k < N; ++k) {
   if(!isVisited[k]){
    setTemp[PartNo-1].push_back(k);
   }
  }
  // puting into the set
  FinalPartPtr.resize(PartNo+1);
  for (int l = 0; l < setTemp.size(); ++l) {
   int ss = setTemp[l].size();
   int ip = FinalPartPtr[l];
   FinalPartPtr[l+1] = ip + ss;
   for (int i = 0; i < ss; ++i) {
    assert(ip < N);
    FinalNodePtr[ip] = setTemp[l][i];
    ip++;
   }
   //assert(ip <= n);
  }
  if(!TargetWeight)
   delete []evenWeight;
  delete []isVisited;
 }


 int LBC(const CSC *G, const CSC *Di, ScheduleParameters* Sp, int LoopId, int HintTotLoops,
         std::vector<std::vector<FusedNode*>>& CurNodeList,
         DAG *OutDag,
         std::vector<int>& VToPart, std::vector<std::vector<std::pair<int,int>>>& PartToCoord
 ){

  int finalLevelNo, partNo;
  int *finaLevelPtr, *finalPartPtr, *finalNodePtr;
  VToPart.resize(G->m);
  std::vector<double> cost(G->m, 1);
  std::vector<int> finalPartPtrVec, finalNodePtrVec(G->m);
  if(G->m < G->nnz){
   get_coarse_levelSet_DAG_CSC_tree(G->m, G->p, G->i,
                                    G->stype,
                                    finalLevelNo,
                                    finaLevelPtr,partNo,
                                    finalPartPtr,finalNodePtr,
                                    Sp->_num_w_partition,Sp->_lbc_agg,
                                    Sp->_lbc_initial_cut, cost.data());
  } else {
   if(Sp->SeedPartitioningParallelism == BFS){
    //print_csc(1, "Di", 6, Di->p, Di->i, Di->x);
    partitionByBfs(Di->m, Di, cost.data(), nullptr, Sp->_num_w_partition, finalPartPtrVec, finalNodePtrVec);
//    for (int i = 0; i < finalPartPtrVec.size(); ++i) {
//      for (int j = finalPartPtrVec[i]; j < finalPartPtrVec[i+1]; ++j) {
//        //VToPart[finalNodePtrVec[j]] = i;
//        std::cout<<finalNodePtrVec[j]<<" ";
//      }
//      std::cout<<"\n";
//    }
   } else { // Consecutive
    for (int i = 0; i < G->m; ++i)
     finalNodePtrVec[i] = i;
    partitionByWeight(G->m, finalNodePtrVec.data(), cost.data(),
                      Sp->_num_w_partition, nullptr, finalPartPtrVec);
   }
   finalLevelNo = 1;
   finaLevelPtr = new int[2];
   finaLevelPtr[0] = 0; finaLevelPtr[1] = finalPartPtrVec.size();
   finalPartPtr = finalPartPtrVec.data();
   finalNodePtr = finalNodePtrVec.data();
  }

  //HDAGG::build_coarsened_level_parallel
  //auto ret_hwf = new FusedNode(G->m, final_level_no, fina_level_ptr, final_part_ptr, final_node_ptr);
  //CurNodeList.resize(HintTotLoops*Sp->_num_w_partition);// allocate for other loops coming
  CurNodeList.resize(finalLevelNo);
  PartToCoord.resize(finalLevelNo);
  std::vector<FusedNode*> wPart; int partCnt=0;
  for (int i = 0; i < finalLevelNo; ++i) {
   //part_to_coord[i].resize(sp->_num_w_partition);
   for (int j = finaLevelPtr[i]; j < finaLevelPtr[i + 1]-1; ++j) {
    int k = finalPartPtr[j];
    int kpOne = finalPartPtr[j + 1];
    auto *curFn = new FusedNode(HintTotLoops, LoopId, kpOne - k, finalNodePtr + k, partCnt);
    CurNodeList[i].push_back(curFn);
    PartToCoord[i].emplace_back(i,j - finaLevelPtr[i]);
    for (int kk = finalPartPtr[j]; kk < finalPartPtr[j + 1]; ++kk) {
     auto curV = finalNodePtr[kk];
     VToPart[curV] = partCnt;
    }
    partCnt++;
   }
  }
  // Let's build the partitioned DAG
  OutDag->build_DAG_from_mapping_CSC(partCnt, Sp->_num_w_partition, VToPart.data(), G);
//  std::vector<bool> destination(part_cnt);
//  for (int i = 0; i < G->m; ++i) {
//   int src = v_to_part[i];
//   for (int j = G->p[i]; j < G->p[i+1]; ++j) {
//    int dst = v_to_part[G->i[j]];
//    out_dag->_part_vertex_bool[src][dst] = true;
//   }
//  }
//
//  for (int i = 0; i < final_level_no; ++i) {
//   for (int j = fina_level_ptr[i]; j < fina_level_ptr[i + 1]; ++j) {
//    std::fill(destination.begin(), destination.end(), 0);
//    int src = cur_node_list[i][j-fina_level_ptr[i]]->_vertex_id;
//    for (int k = final_part_ptr[j]; k < final_part_ptr[j + 1]; ++k) {
//     auto cur_v = v_to_part[final_node_ptr[k]]; //destination partition
//     destination[cur_v] = true;
//    }
//    for (int k = 0; k < destination.size(); ++k) {
//     if(destination[k]){
//      out_dag->add_vertex(src, k);
//     }
//    }
//   }
//  }
  delete []finaLevelPtr;
  if(G->m < G->nnz){
   delete []finalPartPtr;
   delete []finalNodePtr;
  }
  return 0;
 }


 // FIXME: WIP
 void get_iteration_schedule_mapping(int nIterations,
                                     const std::vector<std::vector<FusedNode*>>& cur_node_list,
                                     std::vector<std::pair<int,int>>& iteration_to_part){
  iteration_to_part.resize(nIterations);
  for (int i = 0; i < cur_node_list.size(); ++i) {
   for (int j = 0; j < cur_node_list[i].size(); ++j) {
    for (int k = 0; k < cur_node_list[i][j]->_list.size(); ++k) {
     int cur_v = cur_node_list[i][j]->_list[0][j];
     //iteration_to_part[cur_v] = std::make_tuple(i,j);
    }
   }
  }
 }






} // namespace sym_lib