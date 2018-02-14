
#include "add.h"

/*
  This is the function that each thread will execute on the GPU. The
  fact that it executes on the device is indicated by the __global__
  modifier in front of the return type of the function. After that,
  the signature of the function isn't special - in particular, the
  pointers we pass in should point to memory on the device, but this
  is not indicated by the function's signature.
 */
__global__ void add(int *a, int *b, int *c) {

  /*
    Each thread knows its identity in the system. This identity is
    made available in code via indices blockIdx and threadIdx. We
    write blockIdx.x because block indices are multidimensional. In
    this case, we have linear arrays of data, so we only need one
    dimension. If this doesn't make sense, don't worry - the important
    thing is that the first step in the function is converting the
    thread's indentity into an index into the data.
   */
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  /*
    We make sure that the thread_id isn't too large, and then we
    assign c = a + b using the index we calculated above.

    The big picture is that each thread is responsible for adding one
    element from a and one element from b. Each thread is able to run
    in parallel, so we get speedup.
   */
  if (thread_id < N) {
    c[thread_id] = a[thread_id] + b[thread_id];
  }
}

__global__ void mult(int size,int** a, int** b, int** c) {
	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;
	int x, y;
	/*for(int id_x = blockIdx.x * blockDim.x + threadIdx.x; id_x < size; id_x += stride_x){
		for(int id_y = blockIdx.y * blockDim.y + threadIdx.y; id_y < size;	id_y += stride_y){
			c[id_x][id_y] = a[id_x][id_y] * b[id_y][id_x];
		}
	}*/
	for(int j = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x
			+ (blockIdx.x * blockDim.x + threadIdx.x);
			j < size*size; j += stride_x * stride_y){
		x = j/size;
		y = j%size;
		c[x][y] = a[x][y] * b[y][x];
	}
}
