# Illustration of simple knn algorithm via CUDA C
The algorithm is quite rudimentary with respect to k nearest neighbours (knn). 
The code snippet is in [vanilla 3dgs](https://github.com/graphdeco-inria/gaussian-splatting.git) and many other 3dgs-based repositories e.g. the masterpiece of [SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians.git)
In its applicaton in 3d gaussian splatting, for each point in a vector, a gaussian is created whose scale is based up on its information in terms of distance to its neighbours:
i.e.the average distance of itself and its top three nearest neighbours.
The algorithm is essentially quite convenient for its adaptation to the deploy of GPU computing.

The calling from main fucntion in typical 3dgs repo  is like this:

```python
from simple_knn._C import distCUDA2
```
...
```python
distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda())
```

The result is a vector of length equivalent to the number of points, where each elements is the average distance of itself and its three shortest distance.
There is also independent repo [simple-knn](https://github.com/camenduru/simple-knn.git) which I don't have time to check its relation with the one used in 3dgs.



This repository is created for those who:
-  wants to know what CUDA programming looks like;
-  is curious about the realization in the background of GPU deployment;
-  would like to have detailed picture of the point cloud  manipulation in 3d gaussian splatting.



Therefore, I humbly offer the annotation to the simple_knn algorithm with **Detailed linewise illustration is finished via code annotation, especially in <simple_knn.cu>.**

The following is brief introduction of each file.

## setup.py
This file contains metadata about your project, such as the name, version, author, dependencies, etc. 
It's used by `pip` to install the package.

- ext_modules:
This argument specifies a list of setuptools.


- cmdclass:
This argument allows you to override the default commands used by setuptools during the build and installation process. It's a dictionary where the keys are command names and the values are custom command classes you define.
The most common use case for cmdclass is to override the build_ext command, which is responsible for building the extension modules.


- CUDAExtansion:
specifically designed to build CUDA extensions for PyTorch. It simplifies the process of creating CUDA extensions by providing default settings and handling CUDA-specific compilation details.


- Why .h Files Aren't Explicitly Mentioned in CUDAExtension

While you might be accustomed to explicitly listing header files in compilation processes, the CUDAExtension class in PyTorch handles header files implicitly.

The C++ compiler automatically includes header files based on the #include directives within your source files (.cpp and .cu). This means you don't need to manually list them in the CUDAExtension definition.


## ext.cpp
It exposes the cpp function to python.

## spatial.h & spatial.cu
Declaration and definition of `distCUDA2`.

## simple_knn.h & simple_knn.cu
spatial.h & spatial.cu imports things in simple_knn.h & simple_knn.cu.
spatial.cu is the brains of the repo.
