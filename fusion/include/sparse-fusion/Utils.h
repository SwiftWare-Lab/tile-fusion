//
// Created by Kazem on 2023-05-07.
//

#ifndef SPARSE_FUSION_UTILS_H
#define SPARSE_FUSION_UTILS_H

namespace sym_lib{

 void partitionByWeight(int N,const int *Set, const double *Weight,
                        int NParts,
                        double *TargetWeight,
                        std::vector<int> &Indices);

} // namespace sym_lib
#endif //SPARSE_FUSION_UTILS_H
