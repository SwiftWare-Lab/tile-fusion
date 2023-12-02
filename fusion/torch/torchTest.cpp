//
// Created by salehm32 on 29/11/23.
//

#include "Torch_GCN_Layer_Utils.h"
#include <iostream>

int main() {
//  torch::Tensor tensor = torch::rand({2, 3});
  int *a = new int[1000];
  for (int i = 0; i < 1000; ++i) {
    a[i] = i;
  }
  torch::Tensor tensor = torch::from_blob(a, {40,25}, torch::kInt32);
  std::cout << tensor << std::endl;
}