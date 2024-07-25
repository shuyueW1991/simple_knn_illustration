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

#define BOX_SIZE 1024

#include "cuda_runtime.h"  // obligatory for all pure cuda c codes
#include "device_launch_parameters.h"  
//The header file device_launch_parameters.h is typically used in CUDA C++ programming to provide essential parameters for launching kernels on a GPU. 
//These parameters include grid dimensions and block dimensions, though not found in this code.

#include "simple_knn.h" 
#include <cub/cub.cuh>
// By including cub/cub.cuh, you gain access to all the functionalities provided by the CUB library. This includes a wide range of algorithms and data structures for tasks such as:
// Data movement: Copying, loading, storing, and shuffling data between host and device memory.
// Data reduction: Summation, product, min/max, and other reduction operations.
// Data scanning: Prefix sum (scan) operations.
// Sorting: Radix sort, merge sort, and other sorting algorithms.
// Searching: Binary search and other search operations.
// Histogramming: Counting occurrences of elements in an array.

#include <cub/device/device_radix_sort.cuh>
// radix sorting algorithm is deployed to sort Morton code.
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
// thrust::device_vector thing...
#include <thrust/sequence.h>
// providing indice for avector
#define __CUDACC__
// #define __CUDACC__ is a preprocessor macro specifically defined within the CUDA environment. It serves as a flag to indicate that the code is being compiled by the NVIDIA CUDA compiler (nvcc).

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
// The cooperative_groups namespace in CUDA C++ provides a set of utilities for managing 
// and synchronizing groups of threads within a thread block. 
// It offers a higher-level abstraction compared to the lower-level synchronization primitives like __syncthreads(), 
// making it easier to write correct and efficient parallel code.
// several functions:
	// this_thread(): Returns a handle to the current thread.
	// this_thread_block(): Returns a handle to the current thread block.
	// tiled_partition(): Creates a tiled partition of the thread block, dividing it into smaller groups of threads.
	// group_sync(): Synchronizes all threads within a cooperative group.


struct CustomMin
{
	__device__ __forceinline__   // These qualifiers indicate that the following function will be executed on the GPU and that the compiler should attempt to inline the function for performance optimization.
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
		// calculates the minimum component-wise of the input vectors a and b 
		// and returns a new float3 vector containing the minimum values.
	}
};

struct CustomMax // syntaxically similar to `CustomMin`
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};

__host__ __device__ uint32_t prepMorton(uint32_t x)
// generating Morton codes, a space-filling curve used to map multi-dimensional data into a one-dimensional space. 
// This function specifically prepares a 32-bit integer value for subsequent Morton code generation.
{
	x = (x | (x << 16)) & 0x030000FF; // Interleaves the upper and lower 16 bits of x.
	x = (x | (x << 8)) & 0x0300F00F;  // Interleaves the next 8 bits of x with the previous result.
	x = (x | (x << 4)) & 0x030C30C3;  // Interleaves the next 4 bits of x with the previous result.
	x = (x | (x << 2)) & 0x09249249;  // Interleaves the remaining 2 bits of x with the previous result.

	// Interleaving means taking bits from different positions in a number and placing them alternately in a new number.
	// Imagine you have two numbers, A and B, in binary format:
		// A = 1011
		// B = 0100
	// Interleaving these numbers would result in a new number C:
		// C = 10011010
	// As you can see, the bits from A and B are placed alternately in C.
	// A usuall way of generating morton code a 3D point is like this:
		// uint32_t morton3D(int x, int y, int z) {
    	// return (prepMorton(x) << 0) | (prepMorton(y) << 1) | (prepMorton(z) << 2);
	// }
	// In this example, the morton3D function calls prepMorton for each coordinate, 
	// shifts the results, and combines them to create a final Morton code.

	return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2); // see the annotation in `prepMorton`.
}

__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
// efficiently compute Morton codes for an array of 3D points.
{
	auto idx = cg::this_grid().thread_rank();
	// This line obtains the global thread index within the grid using the cooperative_groups library

	if (idx >= P)
		return;

	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	auto idx = cg::this_grid().thread_rank();
	// There's no need for the more complex cg::this_grid().thread_rank() in this context.

	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}
	// It is not just a redundant way of initizliation by using indices vector. 
	//Matter of fact, the real comparison is done immediately in below... but the thing is, the boxes needs to register the `me` thing in a box. 
	// The box means nothing UNLESS the data is sorted morton codes, which has spatial significance if you think about it: with that, the closer codes corresponds to nearer postion in original data....

	__shared__ MinMax redResult[BOX_SIZE];

	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		__syncthreads();

		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads(); 
		// make sure that the concerned part is always in the first half.
	}

	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
	//  writing the final reduced MinMax value for each block to the output array boxes.
	// since the min- max- thing within the block, the min-max-est thing is now at the first thread of the blcok.
	// technically, the box corresponds to a grid, in the lingo of CUDA programming.
	// Thus, each box element in the box vector registers the `me` thing of this box/grid.
}

__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
// The function efficiently maintains the K smallest distances without sorting the entire list at each update.
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX }; // the `best` variable belongs to a thread now.

	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<3>(point, points[indices[i]], best);
	}
	// now we have found the top 3 minimum distances, among the 3 + 3 distances within neighbouring grids.

	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point); // get the distance to each box from the current point.
		if (dist > reject || dist > best[2])
			continue;

		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<3>(point, points[indices[i]], best); // search the 'better' distance within potential 'good' box.
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}

void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;


	// The subsequent 3 lines demonstrate a two-step approach to perform a reduction using cub::DeviceReduce::Reduce with a custom CustomMin operator.
	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	// When the first argument to cub::DeviceReduce::Reduce is nullptr, the specified reduction operator (like CustomMin) is ignored. 
	// However, the first argument is nullptr, and the actual reduction doesn't occur here.
	// The purpose of this call is to determine the required size (temp_storage_bytes) for the temporary storage needed by the reduction operation. 
	// This size depends on the data type, reduction operator, and other factors.
	thrust::device_vector<char> temp_storage(temp_storage_bytes);
	// allocates the required temporary storage using a thrust::device_vector<char> named temp_storage with a size of temp_storage_bytes bytes. 
	//Note that char is used here for convenience; the specific data type doesn't necessarily matter as long as it's large enough to hold the temporary data.
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	// This time, the reduction operation is performed using the specified arguments, with the actual data for the temporary storage through temp_storage.data().get(). 
	// Only when a non-null pointer to temporary storage is provided in the first argument does the specified reduction operator come into play. 
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);
	//copy the data from device to host.

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	// similar with CustomMin thing above, so no need to create temporary storage again.
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);


	thrust::device_vector<uint32_t> morton(P);
	// Creates a Thrust device vector named morton that can hold `P` elements of type uint32_t. 
	// This vector is used to store Morton codes generated from a set of points.
	thrust::device_vector<uint32_t> morton_sorted(P);
	// Creates a Thrust device vector named morton_sorted that can hold `P` elements of type uint32_t. 
	// This vector is used to store morton_sorted codes that is the radix-sorted Morton codes.	
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get()); // the (P + 256 -1) assures that the block number in a grid is surely at least 1, and the minus one is due to the beginning of a sequence in C/CUDA/C++ is 0.
	// morton.data().get(): This retrieves a pointer to the underlying data storage of the morton device vector.
	// <<block number in a grid, block thd number>>

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	// Initializes the indices vector with sequential values from 0 to P-1. 
	// This effectively creates an array of indices representing the positions of elements in another data structure.
	thrust::device_vector<uint32_t> indices_sorted(P);


	// copy the above two-step reduction aproach.
	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	// This line effectively sorts the Morton codes stored in the morton vector along with their corresponding indices from the indices vector. 
	// The primary target of the radix sort in this specific code is the morton vector of Morton codes, which are essentially interleaved bit representations of 3D coordinates.
	// The sorted results are placed in the morton_sorted and indices_sorted vectors, respectively.
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	// By sorting Morton codes, we effectively order the corresponding points in a space-filling curve, 
	// which helps locate the closest point to a given query point.


	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	boxMeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}