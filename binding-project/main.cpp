//
// Created by salehm32 on 03/07/24.
//

#include "Functions.h"

std::vector<torch::Tensor> inspect(torch::Tensor Adj, int64_t MTileSize);

std::vector<torch::Tensor> inspect_vt(torch::Tensor Adj, int64_t InDim, int64_t OutDim, int64_t CacheSize, int64_t NumThreads);

std::vector<torch::Tensor> executeFusedGeMMSpMM(torch::Tensor Adj, torch::Tensor Weight, torch::Tensor Feature,
                                    torch::Tensor LevelPtr, torch::Tensor MixPtr, torch::Tensor Partition,
                                    int64_t NumThreads);

torch::Tensor fusedGeMMSpMM(torch::Tensor Adj, torch::Tensor Feature,
                                    torch::Tensor Weight, std::vector<torch::Tensor> Schedule,
                                    int64_t NumThreads);

torch::Tensor fusedGeMMSpMM_vt_ro(torch::Tensor Adj, torch::Tensor Feature,
                                 torch::Tensor Weight, std::vector<torch::Tensor> Schedule,
                                 int64_t NumThreads);
torch::Tensor geMMSpMMFusedBackward(torch::Tensor Adj, torch::Tensor Feature,
                                    torch::Tensor Weight, std::vector<torch::Tensor> Schedule,
                                    int64_t NumThreads);
torch::Tensor cachedSpMMGeMM(torch::Tensor AF, torch::Tensor Weight, int64_t NumThreads);

std::vector<torch::Tensor> inspect(torch::Tensor Adj, int64_t MTileSize) {
    std::vector<int *> schedule = createSchedule(Adj.crow_indices().data_ptr<int32_t>(),
                                                 Adj.col_indices().data_ptr<int32_t>(),
                                                 Adj.size(0),
                                                 MTileSize);
    torch::Tensor levelPtr = torch::from_blob(
            schedule[0], {1, 3},
            [](void *Ptr) { delete[] static_cast<int32_t *>(Ptr); }, torch::kInt32);
    torch::Tensor mixPtr = torch::from_blob(
            schedule[1], {1, schedule[3][0]},
            [](void *Ptr) { delete[] static_cast<int32_t *>(Ptr); }, torch::kInt32);
    torch::Tensor partition = torch::from_blob(
            schedule[2], {1, schedule[3][1]},
            [](void *Ptr) { delete[] static_cast<int32_t *>(Ptr); }, torch::kInt32);
    return {levelPtr, mixPtr, partition};
}

std::vector<torch::Tensor> inspect_vt(torch::Tensor Adj, int64_t InDim, int64_t OutDim, int64_t CacheSize, int64_t NumThreads){
    int m = Adj.size(0);
    int nnz = Adj._nnz();
    int** rawSchedule = generateVariableTileSizeScheduleGeMMSpMM(m,
                                                                             Adj.crow_indices().data_ptr<int>(),
                                                                             Adj.col_indices().data_ptr<int>(),
                                                                             InDim, OutDim, CacheSize, NumThreads, sizeof(float));

    int numKernels = 2;
//    int *uFAp = new int[m + 1];
//    int *uFAi = new int[nnz];
//    float *uFAx = new float[nnz];
//    int* l2Ptr = new int[m];
    int* levelPtr = rawSchedule[0];
    int* mixPtr = rawSchedule[1];
    int* id = rawSchedule[2];
    int mixPtrSize = levelPtr[2] * 2 + 2;
//    int newMixPtrSize = (levelPtr[numKernels] + 1) * numKernels;
//    int* newMixPtr = new int[newMixPtrSize];
//    createReorderedAdj(Adj.size(0), nnz, Adj.crow_indices().data_ptr<int>(),
//                       Adj.col_indices().data_ptr<int>(), Adj.values().data_ptr<float>(), levelPtr,
//                       rawSchedule[1], rawSchedule[2], uFAp, uFAi, uFAx, newMixPtr, l2Ptr);
//    delete[] rawSchedule[1];
//    delete[] rawSchedule[2];
//    auto uFApTensor = torch::from_blob(uFAp, {m+1}, [](void *Ptr) { delete[] static_cast<int32_t *>(Ptr); }, torch::kInt32);
//    auto uFAiTensor = torch::from_blob(uFAi, {nnz}, [](void *Ptr) { delete[] static_cast<int32_t *>(Ptr); }, torch::kInt32);
//    auto uFAxTensor = torch::from_blob(uFAx, {nnz}, [](void *Ptr) { delete[] static_cast<float *>(Ptr); }, torch::kFloat32);
//    auto adjTensor = torch::sparse_csr_tensor(uFApTensor, uFAiTensor, uFAxTensor, torch::kFloat32);
    auto levelPtrTensor = torch::from_blob(levelPtr, {3}, [](void *Ptr) { delete[] static_cast<int32_t *>(Ptr); }, torch::kInt32);
    auto mixPtrTensor = torch::from_blob(mixPtr, {mixPtrSize}, [](void *Ptr) { delete[] static_cast<int32_t *>(Ptr); }, torch::kInt32);
    auto l2PtrTensor = torch::from_blob(id, {m}, [](void *Ptr) { delete[] static_cast<int32_t *>(Ptr); }, torch::kInt32);
    return {levelPtrTensor, mixPtrTensor, l2PtrTensor};
}

std::vector<torch::Tensor> executeFusedGeMMSpMM(torch::Tensor Adj, torch::Tensor Weight, torch::Tensor Feature,
                                    torch::Tensor LevelPtr, torch::Tensor MixPtr, torch::Tensor Partition,
                                    int64_t NumThreads) {
    auto start = std::chrono::system_clock::now();
    float *out = new float[Adj.size(0) * Weight.size(1)]{};
    fusedMKLGeMMSpMM(Adj.size(0),
                     Adj.crow_indices().data_ptr<int32_t>(),
                     Adj.col_indices().data_ptr<int32_t>(),
                     Adj.values().data_ptr<float>(),
                     Feature.size(1),
                     Weight.size(1), Feature.data_ptr<float>(),
                     Weight.data_ptr<float>(), out, NumThreads, 2, LevelPtr.data_ptr<int32_t>(),
                     MixPtr.data_ptr<int32_t>(),
                     Partition.data_ptr<int32_t>());
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    auto timeWrapper = torch::tensor(elapsed.count());
//    std::cout << elapsed.count() << '\n';
    return {torch::from_blob(
            out, {Adj.size(0), Weight.size(1)},
            [](void *ptr) { delete[] static_cast<float *>(ptr); }, torch::kFloat32),
            timeWrapper};
}


torch::Tensor fusedGeMMSpMM(torch::Tensor Adj, torch::Tensor Feature,
                                    torch::Tensor Weight, std::vector<torch::Tensor> Schedule,
                                    int64_t NumThreads){
    return FusedGeMMSpMM::apply(Adj, Feature, Weight, Schedule[0], Schedule[1], Schedule[2], NumThreads);
}

torch::Tensor fusedGeMMSpMM_vt_ro(torch::Tensor Adj, torch::Tensor Feature,
                                 torch::Tensor Weight, std::vector<torch::Tensor> Schedule,
                                 int64_t NumThreads){
    return FusedGeMMSpMMROAdj::apply(Adj, Feature, Weight, Schedule[0], Schedule[1], Schedule[2], NumThreads);
}

torch::Tensor geMMSpMMFusedBackward(torch::Tensor Adj, torch::Tensor Feature,
                                   torch::Tensor Weight, std::vector<torch::Tensor> Schedule,
                                   int64_t NumThreads){
    return SGForwardFusedGSBackward::apply(Adj, Feature, Weight, Schedule[0], Schedule[1], Schedule[2], NumThreads);
}

torch::Tensor cachedSpMMGeMM(torch::Tensor AF, torch::Tensor Weight, int64_t NumThreads){
    return ForwardCachingAF::apply(AF, Weight, NumThreads);
}


TORCH_LIBRARY(sw_gcn, m) {
    m.def("inspect", &inspect);
    m.def("inspect_vt_ro", &inspect_vt);
    m.def("executeFusedGeMMSpMM", &executeFusedGeMMSpMM);
    m.def("fusedGeMMSpMM", &fusedGeMMSpMM);
    m.def("fusedGeMMSpMM_vt_ro", &fusedGeMMSpMM_vt_ro);
    m.def("geMMSpMM_f_bw", &geMMSpMMFusedBackward);
    m.def("cachedSpMMGeMM", &cachedSpMMGeMM);
}