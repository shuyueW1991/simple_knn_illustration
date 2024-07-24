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

#include <torch/extension.h> 
//This line includes the necessary header file from the PyTorch C++ API. 
//This header provides essential definitions and functions for interacting with PyTorch from C++.

#include "spatial.h"
// include the file that contains the `disCUDA2` in the following.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // This macro defines a Pybind11 module named TORCH_EXTENSION_NAME 
  // (which is typically replaced by a macro defined in the build script). 
  // The m variable is an object representing the module.

  m.def("distCUDA2", &distCUDA2);
  // This line exposes the C++ function distCUDA2 to Python as a function named distCUDA2. 
  // The &distCUDA2 part passes the address of the function to Pybind11 for binding.

}
