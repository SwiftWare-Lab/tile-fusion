//
// Created by salehm32 on 03/07/24.
//

#include "Functions.h"

std::vector<torch::Tensor> inspect(torch::Tensor Adj, int64_t MTileSize);

std::vector<torch::Tensor> executeFusedGeMMSpMM(torch::Tensor Adj, torch::Tensor Weight, torch::Tensor Feature,
                                    torch::Tensor LevelPtr, torch::Tensor MixPtr, torch::Tensor Partition,
                                    int64_t NumThreads);

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


TORCH_LIBRARY(sw_gcn, m) {
    m.def("inspect", &inspect);
    m.def("executeFusedGeMMSpMM", &executeFusedGeMMSpMM);
}