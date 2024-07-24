/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0); // how many points 

  // Now create a tensor 'means' filled with zeros.

  auto float_opts = points.options().dtype(torch::kFloat32); 
  // points.options() creates a TensorOptions object, which is a builder-style object for configuring tensor properties.
  // kFloat32 is a constant defined within the PyTorch C++ API to represent the float32 data type.

  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());
  // import a class of SimpleKNN from the included 'simple_knn.h' file.
  // contiguous(): Ensures that the tensor is contiguous in memory, which is often required for efficient C++ interactions.

  return means;
}