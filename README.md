# Illustration of simple knn algorithm via CUDA C
## Scenario
- used in [vanilla 3dgs](https://github.com/graphdeco-inria/gaussian-splatting.git)
- the calling form is:
    ```
    from simple_knn._C import distCUDA2
    ...
    distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda())
    ```

- Detailed linewise illustration is finished via code anotation.


# architecture

## setup.py
This file contains metadata about your project, such as the name, version, author, dependencies, etc. 
It's used by pip to install the package.

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
expose the cpp function to python.

## spatial.h & spatial.cu
declaration and definition of `distCUDA2`.

## simple_knn.h & simple_knn.cu
spatial.h & spatial.cu imports things in simple_knn.h & simple_knn.cu.
spatial.cu is the core of the repo.
