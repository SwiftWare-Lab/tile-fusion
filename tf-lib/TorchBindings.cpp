#include "Functions.h"

#include <torch/extension.h>

#include <stdexcept>
#include <vector>

namespace {

torch::Tensor contiguous_float_cpu(torch::Tensor tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
  return tensor.contiguous();
}

torch::Tensor contiguous_int_cpu(torch::Tensor tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  if (tensor.scalar_type() == torch::kInt32) {
    return tensor.contiguous();
  }
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int32 or int64");
  return tensor.to(torch::kInt32).contiguous();
}

void check_csr_tensor(const torch::Tensor& adj) {
  TORCH_CHECK(adj.device().is_cpu(), "Adj must be a CPU tensor");
  TORCH_CHECK(adj.layout() == torch::kSparseCsr, "Adj must be a sparse CSR tensor");
  TORCH_CHECK(adj.scalar_type() == torch::kFloat32, "Adj values must be float32");
}

std::vector<torch::Tensor> schedule_to_torch(int** raw_schedule, int par_ptr_size,
                                             int partition_size) {
  torch::Tensor level_ptr = torch::from_blob(
      raw_schedule[0], {3},
      [](void* ptr) { delete[] static_cast<int*>(ptr); }, torch::kInt32);
  torch::Tensor par_ptr = torch::from_blob(
      raw_schedule[1], {par_ptr_size},
      [](void* ptr) { delete[] static_cast<int*>(ptr); }, torch::kInt32);
  torch::Tensor partition = torch::from_blob(
      raw_schedule[2], {partition_size},
      [](void* ptr) { delete[] static_cast<int*>(ptr); }, torch::kInt32);
  delete[] raw_schedule;
  return {level_ptr, par_ptr, partition};
}

std::vector<torch::Tensor> spmm_spmm_schedule_to_torch(int** raw_schedule,
                                                       int par_ptr_size,
                                                       int partition_size) {
  torch::Tensor level_ptr = torch::from_blob(
      raw_schedule[0], {3},
      [](void* ptr) { delete[] static_cast<int*>(ptr); }, torch::kInt32);
  torch::Tensor par_ptr = torch::from_blob(
      raw_schedule[1], {par_ptr_size},
      [](void* ptr) { delete[] static_cast<int*>(ptr); }, torch::kInt32);
  torch::Tensor partition = torch::from_blob(
      raw_schedule[2], {partition_size},
      [](void* ptr) { delete[] static_cast<int*>(ptr); }, torch::kInt32);
  torch::Tensor par_type = torch::from_blob(
      raw_schedule[3], {partition_size},
      [](void* ptr) { delete[] static_cast<int*>(ptr); }, torch::kInt32);
  delete[] raw_schedule;
  return {level_ptr, par_ptr, partition, par_type};
}

}  // namespace

std::vector<torch::Tensor> torch_inspect(torch::Tensor adj, int64_t m_tile_size) {
  check_csr_tensor(adj);
  torch::Tensor crow = contiguous_int_cpu(adj.crow_indices(), "Adj crow_indices");
  torch::Tensor col = contiguous_int_cpu(adj.col_indices(), "Adj col_indices");
  std::vector<int*> schedule = swiftware::fusion::createSchedule(
      crow.data_ptr<int>(), col.data_ptr<int>(), adj.size(0), m_tile_size);
  torch::Tensor level_ptr = torch::from_blob(
      schedule[0], {3},
      [](void* ptr) { delete[] static_cast<int*>(ptr); }, torch::kInt32);
  torch::Tensor mix_ptr = torch::from_blob(
      schedule[1], {schedule[3][0]},
      [](void* ptr) { delete[] static_cast<int*>(ptr); }, torch::kInt32);
  torch::Tensor partition = torch::from_blob(
      schedule[2], {schedule[3][1]},
      [](void* ptr) { delete[] static_cast<int*>(ptr); }, torch::kInt32);
  delete[] schedule[3];
  return {level_ptr, mix_ptr, partition};
}

std::vector<torch::Tensor> torch_inspect_vt(torch::Tensor adj, int64_t in_dim,
                                            int64_t out_dim,
                                            int64_t cache_size,
                                            int64_t num_threads) {
  check_csr_tensor(adj);
  torch::Tensor crow = contiguous_int_cpu(adj.crow_indices(), "Adj crow_indices");
  torch::Tensor col = contiguous_int_cpu(adj.col_indices(), "Adj col_indices");
  int** raw_schedule = swiftware::fusion::generateVariableTileSizeScheduleGeMMSpMM(
      adj.size(0), crow.data_ptr<int>(), col.data_ptr<int>(), in_dim, out_dim,
      cache_size, num_threads, sizeof(float));
  int par_ptr_size = raw_schedule[0][2] * 2 + 2;
  return schedule_to_torch(raw_schedule, par_ptr_size, adj.size(0));
}

std::vector<torch::Tensor> torch_inspect_spmm_spmm_fixed(torch::Tensor adj,
                                                         int64_t m_tile_size) {
  check_csr_tensor(adj);
  int** raw_schedule =
      swiftware::fusion::createSpMMSpMMFixedSchedule(adj.size(0), m_tile_size);
  int par_ptr_size = raw_schedule[0][2] + 1;
  return spmm_spmm_schedule_to_torch(raw_schedule, par_ptr_size, adj.size(0) * 2);
}

std::vector<torch::Tensor> torch_inspect_spmm_spmm_vt(torch::Tensor adj,
                                                      int64_t feature_dim,
                                                      int64_t cache_size,
                                                      int64_t num_threads) {
  check_csr_tensor(adj);
  torch::Tensor crow = contiguous_int_cpu(adj.crow_indices(), "Adj crow_indices");
  torch::Tensor col = contiguous_int_cpu(adj.col_indices(), "Adj col_indices");
  int** raw_schedule = swiftware::fusion::generateVariableTileSizeScheduleSpMMSpMM(
      adj.size(0), crow.data_ptr<int>(), col.data_ptr<int>(), feature_dim,
      cache_size, num_threads, sizeof(float));
  int par_ptr_size = raw_schedule[0][2] + 1;
  return spmm_spmm_schedule_to_torch(raw_schedule, par_ptr_size, adj.size(0) * 2);
}

torch::Tensor torch_fused_gemm_spmm(torch::Tensor adj, torch::Tensor feature,
                                    torch::Tensor weight,
                                    std::vector<torch::Tensor> schedule,
                                    int64_t num_threads) {
  check_csr_tensor(adj);
  TORCH_CHECK(schedule.size() == 3, "schedule must contain level_ptr, mix_ptr, partition");
  torch::Tensor feature_c = contiguous_float_cpu(feature, "Feature");
  torch::Tensor weight_c = contiguous_float_cpu(weight, "Weight");
  torch::Tensor values = contiguous_float_cpu(adj.values(), "Adj values");
  torch::Tensor crow = contiguous_int_cpu(adj.crow_indices(), "Adj crow_indices");
  torch::Tensor col = contiguous_int_cpu(adj.col_indices(), "Adj col_indices");
  torch::Tensor level_ptr = contiguous_int_cpu(schedule[0], "LevelPtr");
  torch::Tensor mix_ptr = contiguous_int_cpu(schedule[1], "MixPtr");
  torch::Tensor partition = contiguous_int_cpu(schedule[2], "Partition");

  TORCH_CHECK(feature_c.dim() == 2, "Feature must be 2D");
  TORCH_CHECK(weight_c.dim() == 2, "Weight must be 2D with shape [out_dim, in_dim]");
  TORCH_CHECK(feature_c.size(1) == weight_c.size(1), "Feature dim must match Weight dim");

  auto out = torch::empty({adj.size(0), weight_c.size(0)}, feature_c.options());
  out.zero_();
  mkl_set_num_threads(1);
  swiftware::fusion::fusedMKLGeMMSpMMTransposedWeight(
      adj.size(0), crow.data_ptr<int>(), col.data_ptr<int>(),
      values.data_ptr<float>(), feature_c.size(1), weight_c.size(0),
      feature_c.data_ptr<float>(), weight_c.data_ptr<float>(),
      out.data_ptr<float>(), num_threads, 2, level_ptr.data_ptr<int>(),
      mix_ptr.data_ptr<int>(), partition.data_ptr<int>());
  return out;
}

torch::Tensor torch_fused_gemm_spmm_vt(torch::Tensor adj, torch::Tensor feature,
                                       torch::Tensor weight,
                                       std::vector<torch::Tensor> schedule,
                                       int64_t num_threads) {
  check_csr_tensor(adj);
  TORCH_CHECK(schedule.size() == 3, "schedule must contain level_ptr, mix_ptr, partition");
  torch::Tensor feature_c = contiguous_float_cpu(feature, "Feature");
  torch::Tensor weight_c = contiguous_float_cpu(weight, "Weight");
  torch::Tensor values = contiguous_float_cpu(adj.values(), "Adj values");
  torch::Tensor crow = contiguous_int_cpu(adj.crow_indices(), "Adj crow_indices");
  torch::Tensor col = contiguous_int_cpu(adj.col_indices(), "Adj col_indices");
  torch::Tensor level_ptr = contiguous_int_cpu(schedule[0], "LevelPtr");
  torch::Tensor mix_ptr = contiguous_int_cpu(schedule[1], "MixPtr");
  torch::Tensor partition = contiguous_int_cpu(schedule[2], "Partition");

  TORCH_CHECK(feature_c.dim() == 2, "Feature must be 2D");
  TORCH_CHECK(weight_c.dim() == 2, "Weight must be 2D with shape [out_dim, in_dim]");
  TORCH_CHECK(feature_c.size(1) == weight_c.size(1), "Feature dim must match Weight dim");

  auto out = torch::empty({adj.size(0), weight_c.size(0)}, feature_c.options());
  out.zero_();
  mkl_set_num_threads(1);
  swiftware::fusion::fusedMKLGeMMSpMMTransposedWeightVT(
      adj.size(0), crow.data_ptr<int>(), col.data_ptr<int>(),
      values.data_ptr<float>(), feature_c.size(1), weight_c.size(0),
      feature_c.data_ptr<float>(), weight_c.data_ptr<float>(),
      out.data_ptr<float>(), num_threads, 2, level_ptr.data_ptr<int>(),
      mix_ptr.data_ptr<int>(), partition.data_ptr<int>());
  return out;
}

torch::Tensor torch_cached_spmm_gemm(torch::Tensor af, torch::Tensor weight,
                                     int64_t num_threads) {
  torch::Tensor af_c = contiguous_float_cpu(af, "AF");
  torch::Tensor weight_c = contiguous_float_cpu(weight, "Weight");
  TORCH_CHECK(af_c.dim() == 2, "AF must be 2D");
  TORCH_CHECK(weight_c.dim() == 2, "Weight must be 2D with shape [out_dim, in_dim]");
  TORCH_CHECK(af_c.size(1) == weight_c.size(1), "AF dim must match Weight dim");
  auto out = torch::empty({af_c.size(0), weight_c.size(0)}, af_c.options());
  mkl_set_num_threads(num_threads);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, af_c.size(0),
              weight_c.size(0), af_c.size(1), 1.0f, af_c.data_ptr<float>(),
              af_c.size(1), weight_c.data_ptr<float>(), weight_c.size(1), 0.0f,
              out.data_ptr<float>(), weight_c.size(0));
  return out;
}

torch::Tensor torch_fused_spmm_spmm(torch::Tensor adj_a, torch::Tensor adj_b,
                                    torch::Tensor feature,
                                    std::vector<torch::Tensor> schedule,
                                    int64_t num_threads) {
  check_csr_tensor(adj_a);
  check_csr_tensor(adj_b);
  TORCH_CHECK(schedule.size() == 4,
              "schedule must contain level_ptr, par_ptr, partition, par_type");
  torch::Tensor feature_c = contiguous_float_cpu(feature, "Feature");
  TORCH_CHECK(feature_c.dim() == 2, "Feature must be 2D");
  TORCH_CHECK(feature_c.size(1) % 32 == 0,
              "Feature dim must be a multiple of 32 for the AVX256 SpMM-SpMM kernel");

  torch::Tensor a_values = contiguous_float_cpu(adj_a.values(), "A values");
  torch::Tensor a_crow = contiguous_int_cpu(adj_a.crow_indices(), "A crow_indices");
  torch::Tensor a_col = contiguous_int_cpu(adj_a.col_indices(), "A col_indices");
  torch::Tensor b_values = contiguous_float_cpu(adj_b.values(), "B values");
  torch::Tensor b_crow = contiguous_int_cpu(adj_b.crow_indices(), "B crow_indices");
  torch::Tensor b_col = contiguous_int_cpu(adj_b.col_indices(), "B col_indices");
  torch::Tensor level_ptr = contiguous_int_cpu(schedule[0], "LevelPtr");
  torch::Tensor par_ptr = contiguous_int_cpu(schedule[1], "ParPtr");
  torch::Tensor partition = contiguous_int_cpu(schedule[2], "Partition");
  torch::Tensor par_type = contiguous_int_cpu(schedule[3], "ParType");

  auto out = torch::empty({adj_b.size(0), feature_c.size(1)}, feature_c.options());
  auto intermediate = torch::empty({adj_a.size(0), feature_c.size(1)}, feature_c.options());
  out.zero_();
  intermediate.zero_();
  swiftware::fusion::fusedSpMMSpMMVectorizedAvx256SP(
      adj_a.size(0), feature_c.size(1), a_crow.data_ptr<int>(),
      a_col.data_ptr<int>(), a_values.data_ptr<float>(), b_crow.data_ptr<int>(),
      b_col.data_ptr<int>(), b_values.data_ptr<float>(),
      feature_c.data_ptr<float>(), out.data_ptr<float>(),
      intermediate.data_ptr<float>(), 2, level_ptr.data_ptr<int>(),
      par_ptr.data_ptr<int>(), partition.data_ptr<int>(),
      par_type.data_ptr<int>(), num_threads);
  return out;
}

void bind_torch_functions(pybind11::module_& m) {
  m.def("torch_inspect", &torch_inspect);
  m.def("torch_inspect_vt", &torch_inspect_vt);
  m.def("torch_inspect_spmm_spmm_fixed", &torch_inspect_spmm_spmm_fixed);
  m.def("torch_inspect_spmm_spmm_vt", &torch_inspect_spmm_spmm_vt);
  m.def("torch_fused_gemm_spmm", &torch_fused_gemm_spmm);
  m.def("torch_fused_gemm_spmm_vt", &torch_fused_gemm_spmm_vt);
  m.def("torch_cached_spmm_gemm", &torch_cached_spmm_gemm);
  m.def("torch_fused_spmm_spmm", &torch_fused_spmm_spmm);
}
