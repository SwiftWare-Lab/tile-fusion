//
// Created by Kazem on 2023-02-18.
//
#include "sparse-fusion/Fusion_Utils.h"
#include "aggregation/def.h"
#include "aggregation/lbc.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "aggregation/test_utils.h"
#include <argparse/argparse.hpp>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>

#ifdef METIS
#include "aggregation/metis_interface.h"
#include "sparse-fusion/SparseFusion.h"
#endif

namespace sym_lib {

argparse::ArgumentParser* addArguments() {
  argparse::ArgumentParser* program = new argparse::ArgumentParser("fusion");
  program->add_argument("-sm", "--sparse-matrix-path")
      .help("specify sparse matrix path");

  program->add_argument("-nt", "--num-threads")
      .default_value(3)
      .help("specify number of threads to be used by kernels.")
      .scan<'d', int>();
  //  program->add_argument("-lc, --use-level-coarsening")
  //      .default_value(1)
  //      .help("use level coarsening")
  //      .scan<'i', int>();
  //  program->add_argument("-la", "--lbc-agg")
  //      .default_value(1)
  //      .help("lbc agg")
  //      .scan<'i', int>();
  program->add_argument("-fm", "--feature-matrix-path")
      .help("Specify feature matrix when applicable.");

  program->add_argument("-rm", "--result-matrix-path")
      .help("Specify result matrix when Applicable");

  program->add_argument("-w1", "--weight1-matrix-path")
      .help("Specify weight matrix for layer 1 when applicable");

  program->add_argument("-w2", "--weight2-matrix-path")
      .help("Specify weight matrix for layer 1 when applicable");
  program->add_argument("-dp", "--data-path")
      .help("specify data path for end to end experiments");

  program->add_argument("-ah", "--add-header")
      .default_value(false)
      .implicit_value(true)
      .help("Specify whether to add the CSV header or not.");

  program->add_argument("-bc", "--b-coloums")
      .default_value(4)
      .help("Specify number of dense matrix columns where applicable.")
      .scan<'i', int>();

  program->add_argument("-ip", "--iter-per-partition")
      .help("Iter per partition")
      .scan<'i', int>();

  program->add_argument("-tn", "--tile-n")
      .help("Specify number of tiles for N.")
      .scan<'i', int>();

  program->add_argument("-sr", "--sampling-ratio")
      .default_value(float(0.2))
      .help("Specify ratio of sampling")
      .scan<'g', float>();

  program->add_argument("-en", "--experiment-name")
      .default_value("gcnFusedSequential")
      .help("Specify the experiment");

  program->add_argument("-ed", "--embed-dim")
      .default_value(10)
      .help("Specify the embedding dimensions.")
      .scan<'i', int>();
  program->add_argument("-mw", "--min-workload-size")
      .default_value(10)
      .help("Specify the minimum workload size to run in parallel in GCNLayerTiledFusedCSCCombined")
      .scan<'i', int>();

  return program;
}

void parse_args(const int Argc, const char **Argv, ScheduleParameters *Sp,
                TestParameters *Tp) {
  argparse::ArgumentParser* program = addArguments();
  try {
    program->parse_args(Argc, Argv); // Example: ./main -abc 1.95 2.47
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }
  if(program->is_used("-dp")){
    Tp->e2e_data_path = program->get("-dp");
    Tp->_mode = "MTX";
    Tp->_matrix_name =
        Tp->e2e_data_path.substr(Tp->e2e_data_path.find_last_of("/\\") + 1);
    Tp->_matrix_path = Tp->e2e_data_path + "/" + Tp->_matrix_name + ".mtx";
    Tp->_gnn_parameters_mode = "MTX";
  }
  else if (!program->is_used("-sm")) {
    Tp->_mode = "Random";
    Tp->_dim1 = Tp->_dim2 = 10;
    Tp->_dim1 = Tp->_dim2 = 10;
    Tp->_matrix_name = "Random_" + std::to_string(Tp->_dim1);
    Tp->_order_method = SYM_ORDERING::NONE;
    Tp->print_header = true;
    Sp->_num_w_partition = Sp->_num_threads;
    Sp->_lbc_agg = Sp->_lbc_initial_cut = 4;
  }
  else {
    Tp->_gnn_parameters_mode= "Random";
    Tp->_matrix_path = program->get("-sm");
    Tp->_mode = "MTX";
    Tp->_matrix_name =
        Tp->_matrix_path.substr(Tp->_matrix_path.find_last_of("/\\") + 1);
  };
  Sp->_num_threads = program->get<int>("-nt");
  int useLevelCoarsening = 1;
  useLevelCoarsening = 4;
  Sp->_lbc_agg = 4;
//  if (auto featMtxPath = program->present("-fm")) {
//    Tp->_gnn_parameters_mode = "MTX";
//    Tp->_feature_matrix_path = featMtxPath.value();
//  } else {
//    Tp->_gnn_parameters_mode = "Random";
//  }
//  if (auto weightMtxPath = program->present("-w1")) {
//    Tp->_gnn_parameters_mode = "MTX";
//    Tp->_weight1_matrix_path = weightMtxPath.value();
//  } else {
//    Tp->_gnn_parameters_mode = "Random";
//  }
//  if (auto weightMtxPath = program->present("-w2")) {
//    Tp->_gnn_parameters_mode = "MTX";
//    Tp->_weight2_matrix_path = weightMtxPath.value();
//  } else {
//    Tp->_gnn_parameters_mode = "Random";
//  }
//  if (auto resultMtxPath = program->present("-rm")) {
//    Tp->_gnn_parameters_mode = "MTX";
//    Tp->_weight2_matrix_path = resultMtxPath.value();
//  } else {
//    Tp->_gnn_parameters_mode = "Random";
//  }
  Tp->_sampling_ratio = program->get<float>("-sr");
  Tp->expariment_name = program->get("-en");

  Tp->print_header = 0;
  if (program->is_used("-ah"))
    Tp->print_header = 1;

  Tp->_b_cols = program->get<int>("-bc");
  Tp->_embed_dim = program->get<int>("-ed");
  Sp->_min_workload_size = program->get<int>("-mw");
  if (auto iterPerPart = program->present<int>("-ip"))
    Sp->IterPerPartition = iterPerPart.value();
  if (auto tileN = program->present<int>("-tn"))
    Sp->TileN = tileN.value();

  delete program;

}

int get_reorderd_matrix(const CSC *L1_csc, std::vector<CSC *> &mat_vec) {
  /// Re-ordering L matrix
#ifdef METIS
  int *perm = NULLPNTR;
  CSC *Lt_ordered_csc;
  CSC *L_ordered_csc;
  // We only reorder L since dependency matters more in l-solve.
  // perm = new int[n]();
  CSC *L1_csc_full = make_full((CSC *)L1_csc);
  metis_perm_general(L1_csc_full, perm);
  auto L1_csc_half =
      make_half(L1_csc_full->n, L1_csc_full->p, L1_csc_full->i, L1_csc_full->x);
  Lt_ordered_csc = transpose_symmetric((CSC *)L1_csc_half, perm);
  L_ordered_csc = transpose_symmetric(Lt_ordered_csc, NULLPNTR);
  mat_vec.push_back(Lt_ordered_csc);
  mat_vec.push_back(L_ordered_csc);
  delete L1_csc_full;
  delete L1_csc_half;
  delete[] perm;
  return 0;
#endif
  return -1;
}

sym_lib::CSR *readSparseMatrix(const std::string &Path) {
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
  while (file.get(next)) {
    if (next == '\n')
      break;
  }

  // Read in col_indices
  for (int i = 0; i < csr->nnz; i++) {
    file >> csr->i[i];
  }
  for (int i = 0; i < csr->nnz; i++) {
    csr->x[i] = 1.0f;
  }

  return csr;
}

CSC *get_matrix_from_parameter(const TestParameters *TP, bool AddSelfLoops) {
  CSC *aCSC = NULLPNTR;
  if (TP->_mode == "Random") {
    aCSC = random_square_sparse(TP->_dim1, TP->_density);
  } else {
    std::string fileExt =
        TP->_matrix_path.substr(TP->_matrix_path.find_last_of(".") + 1);
    if (fileExt == "smtx") {
      auto *tmpCsr = readSparseMatrix(TP->_matrix_path);
      aCSC = csr_to_csc(tmpCsr);
      delete tmpCsr;
    } else if (fileExt == "mtx") {
      std::ifstream fin(TP->_matrix_path);
      sym_lib::read_mtx_csc_real(fin, aCSC, false);
    } else {
      std::cout << "File extension is not supported" << std::endl;
      exit(1);
    }
  }
  return aCSC;
}

Dense *get_dense_matrix_from_parameter(const TestParameters *Tp, int Rows, int Cols, std::string MtxPath) {
  Dense *featureMatrix = NULLPNTR;
  if (Tp->_gnn_parameters_mode == "Random") {
    featureMatrix = random_dense_matrix(Rows, Cols);
  } else {
    //  std::string fileExt = Tp->_feature_matrix_path.substr(
    //      Tp->_matrix_path.find_last_of(".") + 1);
    //  if (fileExt == "mtx") {
    std::ifstream fin(MtxPath);
    sym_lib::read_mtx_array_real(fin, featureMatrix);
    //  } else{
    //   std::cout << "File extension is not supported" << std::endl;
    //   exit(1);
    //  }
  }
  return featureMatrix;
}

Dense *identity_dense_matrix(int M) {
  Dense *denseMatrix = new Dense(M, M, M);
  double *a = denseMatrix->a;
  for (int i = 0; i < M; i++) {
    a[i * M + i] = 1;
  }
  return denseMatrix;
}

Dense *random_dense_matrix(int M, int N) {
  Dense *denseMatrix = new Dense(M, N, N);
  std::random_device randDev;
  std::mt19937 generator(randDev());
  std::uniform_real_distribution<double> distr(-1., 1.);
  double *a = denseMatrix->a;
  for (int i = 0; i < M * N; i++) {
    a[i] = distr(generator);
  }
  return denseMatrix;
}

// starts from in_set in G1 and reaches to all unvisited vertices in G2
// G1 -> G2 , D is transpose of dependence works as a func from G2 to G1
// visited_g1 shows visited vertices of previous coarsened wf
void forward_pairing(CSC *G2, CSC *D, const std::vector<int> &InSet,
                     std::vector<int> &OutSet,
                     const std::vector<bool> &VisitedG1,
                     const std::vector<bool> &VisitedG2) {
  std::vector<bool> tmpG1(G2->m, false);
  for (int id : InSet) {
    tmpG1[id] = true; // each vertex in in_set is visited locally
  }
  for (int id : InSet) {
    // visited_g2[id] = true; // each node is visited itself
    // tmpG1[id] = true; // each vertex in in_set is visited locally
    bool selfDisjoint = true;
    // check all dependent G2 vertices are there
    for (int j = G2->p[id]; j < G2->p[id + 1]; ++j) {
      auto reachedIter = G2->i[j];
      if (!VisitedG2[reachedIter] && reachedIter != id) {
        selfDisjoint = false;
        break;
      }
    }
    if (selfDisjoint) {
      // check all dependent G1 vertices are there
      for (int j = D->p[id]; j < D->p[id + 1]; ++j) {
        auto reachedIter = D->i[j]; // an iteration of G1
        // if(!VisitedG1[reachedIter]){
        if (!tmpG1[reachedIter]) {
          selfDisjoint = false;
          break;
        }
      }
      if (selfDisjoint)
        OutSet.push_back(id); // id can be added now
    }
  }
}

/// Calculates the summation of a vector
/// \tparam type
/// \param n
/// \param vec
/// \return
template <class type> type sumVector(int N, type *Vec) {
  double sum = 0;
  for (int i = 0; i < N; ++i)
    sum += Vec[i];
  return sum;
}

/// Partitions the set into n_parts based on the weight of each element
void partitionByWeight(int N, const int *Set, const double *Weight, int NParts,
                       double *TargetWeight, std::vector<int> &Indices) {
  double *evenWeight;
  if (TargetWeight)
    evenWeight = TargetWeight;
  else {
    evenWeight = new double[NParts];
    double evenW = std::ceil(sumVector(N, Weight) / NParts);
    std::fill_n(evenWeight, NParts, evenW);
  }
  Indices.resize(NParts + 1);
  int j = 0;
  for (int i = 0; i < NParts; ++i) {
    double cWgt = 0;
    while (cWgt < evenWeight[i] && j < N) {
      int cN = Set[j];
      cWgt += Weight[cN];
      j++;
    }
    Indices[i + 1] = j;
  }
  if (!TargetWeight)
    delete[] evenWeight;
}

void partitionByBfs(int N, const CSC *Df, const double *Weight,
                    double *TargetWeight, int PartNo,
                    std::vector<int> &FinalPartPtr,
                    std::vector<int> &FinalNodePtr) {
  double *evenWeight;
  // FinalLevelNo = 1;
  // FinaLevelPtr = new int[FinalLevelNo+1]();
  // FinaLevelPtr[FinalLevelNo] = PartNo;
  // FinalPartPtr = new int[PartNo+1]();
  // FinalNodePtr = new int[N]();
  if (TargetWeight)
    evenWeight = TargetWeight;
  else {
    evenWeight = new double[PartNo];
    double evenW = std::ceil(sumVector(N, Weight) / PartNo);
    std::fill_n(evenWeight, PartNo, evenW);
  }
  auto *isVisited = new bool[N]();
  int visitedNodes = 0;
  int nxtStart = 0;
  std::vector<std::vector<int>> setTemp;
  setTemp.resize(PartNo);
  std::vector<int> stack;
  for (int i = 0; i < PartNo && visitedNodes < N; ++i) {
    assert(stack.size() < N);
    double cWgt = 0;
    while (cWgt < evenWeight[i] && visitedNodes < N) {
      if (stack.empty()) {
        nxtStart = -1;
        for (int k = 0; k < N; ++k) {
          if (!isVisited[k]) {
            nxtStart = k;
            break;
          }
        }
        if (nxtStart < 0)
          break;
        stack.push_back(nxtStart);
        isVisited[nxtStart] = true;
      }
      // do bfs per node nxt_start
      while (!stack.empty()) {
        int cn = stack[0];
        stack.erase(stack.begin());
        cWgt += Weight[cn];
        setTemp[i].push_back(cn);
        visitedNodes++;
        for (int k = Df->p[cn]; k < Df->p[cn + 1]; ++k) {
          auto cnn = Df->i[k];
          if (!isVisited[cnn]) {
            stack.push_back(cnn);
            isVisited[cnn] = true;
            for (int j = Df->p[cnn]; j < Df->p[cnn + 1]; ++j) {
              auto newCnn = Df->i[j];
              if (!isVisited[newCnn]) {
                stack.push_back(newCnn);
                isVisited[newCnn] = true;
              }
            }
          }
        }
        if (cWgt >= evenWeight[i])
          break;
      }
    }
  }
  // if anything is remained add it to last part
  for (int m = 0; m < stack.size(); ++m) {
    setTemp[PartNo - 1].push_back(stack[m]);
  }
  for (int k = 0; k < N; ++k) {
    if (!isVisited[k]) {
      setTemp[PartNo - 1].push_back(k);
    }
  }
  // puting into the set
  FinalPartPtr.resize(PartNo + 1);
  for (int l = 0; l < setTemp.size(); ++l) {
    int ss = setTemp[l].size();
    int ip = FinalPartPtr[l];
    FinalPartPtr[l + 1] = ip + ss;
    for (int i = 0; i < ss; ++i) {
      assert(ip < N);
      FinalNodePtr[ip] = setTemp[l][i];
      ip++;
    }
    // assert(ip <= n);
  }
  if (!TargetWeight)
    delete[] evenWeight;
  delete[] isVisited;
}

int LBC(const CSC *G, const CSC *Di, ScheduleParameters *Sp, int LoopId,
        int HintTotLoops, std::vector<std::vector<FusedNode *>> &CurNodeList,
        DAG *OutDag, std::vector<int> &VToPart,
        std::vector<std::vector<std::pair<int, int>>> &PartToCoord) {

  int finalLevelNo, partNo;
  int *finaLevelPtr, *finalPartPtr, *finalNodePtr;
  VToPart.resize(G->m);
  std::vector<double> cost(G->m, 1);
  std::vector<int> finalPartPtrVec, finalNodePtrVec(G->m);
  if (G->m < G->nnz) {
    get_coarse_levelSet_DAG_CSC_tree(
        G->m, G->p, G->i, G->stype, finalLevelNo, finaLevelPtr, partNo,
        finalPartPtr, finalNodePtr, Sp->_num_w_partition, Sp->_lbc_agg,
        Sp->_lbc_initial_cut, cost.data());
  } else {
    if (Sp->SeedPartitioningParallelism == BFS) {
      // print_csc(1, "Di", 6, Di->p, Di->i, Di->x);
      partitionByBfs(Di->m, Di, cost.data(), nullptr, Sp->_num_w_partition,
                     finalPartPtrVec, finalNodePtrVec);
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
    finaLevelPtr[0] = 0;
    finaLevelPtr[1] = finalPartPtrVec.size();
    finalPartPtr = finalPartPtrVec.data();
    finalNodePtr = finalNodePtrVec.data();
  }

  // HDAGG::build_coarsened_level_parallel
  // auto ret_hwf = new FusedNode(G->m, final_level_no, fina_level_ptr,
  // final_part_ptr, final_node_ptr);
  // CurNodeList.resize(HintTotLoops*Sp->_num_w_partition);// allocate for other
  // loops coming
  CurNodeList.resize(finalLevelNo);
  PartToCoord.resize(finalLevelNo);
  std::vector<FusedNode *> wPart;
  int partCnt = 0;
  for (int i = 0; i < finalLevelNo; ++i) {
    // part_to_coord[i].resize(sp->_num_w_partition);
    for (int j = finaLevelPtr[i]; j < finaLevelPtr[i + 1] - 1; ++j) {
      int k = finalPartPtr[j];
      int kpOne = finalPartPtr[j + 1];
      auto *curFn = new FusedNode(HintTotLoops, LoopId, kpOne - k,
                                  finalNodePtr + k, partCnt);
      CurNodeList[i].push_back(curFn);
      PartToCoord[i].emplace_back(i, j - finaLevelPtr[i]);
      for (int kk = finalPartPtr[j]; kk < finalPartPtr[j + 1]; ++kk) {
        auto curV = finalNodePtr[kk];
        VToPart[curV] = partCnt;
      }
      partCnt++;
    }
  }
  // Let's build the partitioned DAG
  OutDag->build_DAG_from_mapping_CSC(partCnt, Sp->_num_w_partition,
                                     VToPart.data(), G);
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
  delete[] finaLevelPtr;
  if (G->m < G->nnz) {
    delete[] finalPartPtr;
    delete[] finalNodePtr;
  }
  return 0;
}

// FIXME: WIP
void get_iteration_schedule_mapping(
    int nIterations, const std::vector<std::vector<FusedNode *>> &cur_node_list,
    std::vector<std::pair<int, int>> &iteration_to_part) {
  iteration_to_part.resize(nIterations);
  for (int i = 0; i < cur_node_list.size(); ++i) {
    for (int j = 0; j < cur_node_list[i].size(); ++j) {
      for (int k = 0; k < cur_node_list[i][j]->_list.size(); ++k) {
        int cur_v = cur_node_list[i][j]->_list[0][j];
        // iteration_to_part[cur_v] = std::make_tuple(i,j);
      }
    }
  }
}

int MakePartitionIndependent(
    int LNo, int PartNo, int SrcLoopID, int DstLoopID, const CSC *Dm,
    const std::vector<std::vector<FusedNode *>> &FinalNodeList,
    std::vector<int> &ReachedItersList) {
  int height = FinalNodeList.size();
  if (height <= LNo)
    return 0;
  // for (int j = 0; j < FinalNodeList[LNo].size(); ++j) {
  // std::vector<int> reachedItersList;
  int j = PartNo;
  for (int k = 0; k < FinalNodeList[LNo][j]->_list[SrcLoopID].size(); ++k) {
    auto srcIter = FinalNodeList[LNo][j]->_list[SrcLoopID][k];
    for (int cc = 0, jj = Dm->p[srcIter]; jj < Dm->p[srcIter + 1]; ++cc, ++jj) {
      auto reachedIter = Dm->i[jj];
      ReachedItersList.push_back(reachedIter);
    }
  }
  std::sort(ReachedItersList.begin(), ReachedItersList.end());
  auto last = std::unique(ReachedItersList.begin(), ReachedItersList.end());
  ReachedItersList.erase(last, ReachedItersList.end());
  // FinalNodeList[LNo][j]->_list[DstLoopID] = reachedItersList;
  //}
  return 1;
}

void BalanceWithRedundantComputation(
    const std::vector<std::vector<FusedNode *>> &FinalNodeList,
    std::vector<std::vector<FusedNode *>> &UpdatedNodeList, const CSC *Dm,
    double BalancedRatio = 0.5) {
  // specify how many partitions should become self-sufficient
  int numShiftedPartitions = 0;
  for (int i = 0; i < FinalNodeList.size(); ++i) {
    numShiftedPartitions += FinalNodeList[i].size();
  }
  numShiftedPartitions /= FinalNodeList.size();
  //  if(FinalNodeList[0].size() < numShiftedPartitions)
  //   numShiftedPartitions = numShiftedPartitions - FinalNodeList[0].size();
  //  else
  //    numShiftedPartitions = 0;
  numShiftedPartitions *= BalancedRatio; // numShiftedPartitions=0;

  // go over second wavefront and compute the amount redundant computations
  // needed
  std::vector<std::pair<int, double>> redundantComputation;
  int lNo = 1, dstLoop = 0;
  for (int i = 0; i < FinalNodeList[lNo].size(); ++i) {
    std::vector<int> reachedItersList;
    MakePartitionIndependent(lNo, i, 1, dstLoop, Dm, FinalNodeList,
                             reachedItersList);
    int numRedundantComputation = reachedItersList.size();
    redundantComputation.emplace_back(i, numRedundantComputation);
  }

  // pick numShiftedPartitions partitions with the least redundant computations
  std::sort(redundantComputation.begin(), redundantComputation.end(),
            [](const std::pair<int, double> &A, const std::pair<int, double> &B)
                -> bool { return A.second < B.second; });
  // copy the first wavefront from FinalNodeList to UpdatedNodeList
  for (int i = 0; i < FinalNodeList[0].size(); ++i) {
    auto *curNode = new FusedNode(*FinalNodeList[0][i]);
    UpdatedNodeList[0].push_back(curNode);
  }
  for (int i = 0; i < numShiftedPartitions; ++i) {
    auto minPart = redundantComputation[i].first;
    std::vector<int> reachedItersList;
    MakePartitionIndependent(lNo, minPart, 1, 0, Dm, FinalNodeList,
                             reachedItersList);
    auto *curNode = new FusedNode(*FinalNodeList[lNo][minPart]);
    curNode->_list[dstLoop] = reachedItersList;
    UpdatedNodeList[0].push_back(curNode);
  }
  // copy the rest of the wavefronts from FinalNodeList to UpdatedNodeList
  for (int i = numShiftedPartitions; i < FinalNodeList[lNo].size(); ++i) {
    auto minPart = redundantComputation[i].first;
    auto *curNode = new FusedNode(*FinalNodeList[lNo][minPart]);
    UpdatedNodeList[lNo].push_back(curNode);
  }
}

void measureRedundancy(
    sym_lib::CSC *Gi, sym_lib::SparsityProfileInfo &Spi,
    const std::vector<std::vector<FusedNode *>> &FinalNodeList) {

  int totalNode = 0, height = FinalNodeList.size(), width = 0, loopNo = 0;
  // calculate the number of loops in the schedule
  for (int i = 0; i < FinalNodeList.size(); ++i) {
    for (int j = 0; j < FinalNodeList[i].size(); ++j) {
      loopNo = std::max(loopNo, (int)FinalNodeList[i][j]->_list.size());
    }
  }
  std::vector<std::vector<int>> iterCount(loopNo);

  for (int i = 0; i < FinalNodeList.size(); ++i) {
    for (int j = 0; j < FinalNodeList[i].size(); ++j) {
      for (int k = 0; k < FinalNodeList[i][j]->_list.size(); ++k) { // loop id
        totalNode += FinalNodeList[i][j]->_list[k].size();
        // copy iterations of loop k to iterCount
        iterCount[k].insert(iterCount[k].end(),
                            FinalNodeList[i][j]->_list[k].begin(),
                            FinalNodeList[i][j]->_list[k].end());
      }
    }
  }
  // calculate the number of redundant iterations
  int allIterations = 0, uniqueIterations = 0;
  for (int i = 0; i < loopNo; ++i) {
    allIterations += iterCount[i].size();
    std::sort(iterCount[i].begin(), iterCount[i].end());
    auto it = std::unique(iterCount[i].begin(), iterCount[i].end());
    iterCount[i].resize(std::distance(iterCount[i].begin(), it));
    uniqueIterations += iterCount[i].size();
  }
  Spi.RedundantIterations = allIterations - uniqueIterations;
  Spi.UniqueIterations = uniqueIterations;
}

} // namespace sym_lib