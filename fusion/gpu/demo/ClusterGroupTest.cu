//
// Created by salehm32 on 20/06/24.
//
#include <cuda_runtime.h>
#include <cooperative_groups.h>
//Doesn't work on Ampere architecture. If we could have the resource, we should try testing it and using it on Hopper arch.
namespace cg = cooperative_groups;
__global__ void __cluster_dims__(8,1,1) test_cluster_group(){
  cg::cluster_group g = cg::this_cluster();
  printf("Cluster Block: %d --- Cluster Thread: %d\n", g.block_rank(), g.thread_rank());
  printf("Cluster Block: %d --- Cluster Thread: %d\n", g.block_index().x, g.thread_index().x);

}

int main(){
  test_cluster_group<<<8,8>>>();
  cudaDeviceSynchronize();
}