/*
  This program demonstrates the basics of working with cuda. We use
  the GPU to add two arrays. We also introduce cuda's approach to
  error handling and timing using cuda Events.

  This is the main program. You should also look at the header add.h
  for the important declarations, and then look at add.cu to see how
  to define functions that execute on the GPU.
 */

#include <iostream>

#include "add.h"

int main() {
  int matSize = 1000;
  int sequential = 1;
  int blocks = 1;
  int threads = 1;

  //get array dimensions
  std::cout << "Please enter the dimensions of the matrix (1000<=matSize<=10000):";
  std::cin >> matSize;
  //std::cout << "you input: " << matSize << std::endl;

  //get if we are using cuda or sequential addtion
  std::cout << "Sequential or CUDA?(1=Sequential, 0=CUDA):";
  std::cin >> sequential;

  if(sequential < 1){
    std::cout << "Please enter the number of blocks to be used:";
    std::cin >> blocks;
    if(blocks < 1 ){//|| blocks > 256){
    	std::cout << "invalid block number, using default of 256 (Max 65535)." << std::endl;
    	blocks = matSize*matSize;
    }

    std::cout << "Please enter the number of threads per block:";
    std::cin >> threads;
    if(threads < 1 ){//|| threads > 32){
    	std::cout << "invalid thread number, using default of 32 (Max 65535)." << std::endl;
    	threads = 32;
    }
    /*if(blocks*threads != matSize*matSize){
    	std::cout << "insufficient blocks and threads used, switching to default." << std::endl;
    	blocks = matSize*matSize;
    	threads = 1;
    }*/
  }

  //int* a[matSize];
  //int* b[matSize];
  //int* c[matSize];

  /*
    These will point to memory on the GPU - notice the correspondence
    between these pointers and the arrays declared above.
   */
  //int (*dev_a)[matSize], (*dev_b)[matSize], (*dev_c)[matSize];
  int **dev_a, **dev_b, **dev_c;

  /*
    These calls allocate memory on the GPU (also called the
    device). This is similar to C's malloc, except that instead of
    directly returning a pointer to the allocated memory, cudaMalloc
    returns the pointer through its first argument, which must be a
    void**. The second argument is the number of bytes we want to
    allocate.

    NB: the return value of cudaMalloc (like most cuda functions) is
    an error code. Strictly speaking, we should check this value and
    perform error handling if anything went wrong. We do this for the
    first call to cudaMalloc so you can see what it looks like, but
    for all other function calls we just point out that you should do
    error checking.

    Actually, a good idea would be to wrap this error checking in a
    function or macro, which is what the Cuda By Example book does.
   */
  //std::cout << "1" << std::endl;
  cudaError_t err = cudaMallocManaged( (void**) &dev_a, (matSize*matSize) * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  cudaMallocManaged( (void**) &dev_b, (matSize*matSize) * sizeof(int));
  cudaMallocManaged( (void**) &dev_c, (matSize*matSize) * sizeof(int));

  // These lines just fill the host arrays with some data so we can do
  // something interesting. Well, so we can add two arrays.
  //std::cout << "2" << std::endl;
  for(int iter = 0; iter<matSize;iter++){
	  cudaMallocManaged( (void**) &(dev_a[iter]), (matSize)*sizeof(int));
	  cudaMallocManaged( (void**) &(dev_b[iter]), (matSize)*sizeof(int));
	  cudaMallocManaged( (void**) &(dev_c[iter]), (matSize)*sizeof(int));
  	  for(int cur = 0; cur<matSize;cur++){
  		dev_a[iter][cur] = iter*cur;
  		dev_b[iter][cur] = iter*cur;
  		dev_c[iter][cur] = 0;
  	  }
    }
  //std::cout << "3" << std::endl;

  /*
    The following code is responsible for handling timing for code
    that executes on the GPU. The cuda approach to this problem uses
    events. For timing purposes, an event is essentially a point in
    time. We create events for the beginning and end points of the
    process we want to time. When we want to start timing, we call
    cudaEventRecord.

    In this case, we want to record the time it takes to transfer data
    to the GPU, perform some computations, and transfer data back.
  */
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord( start, 0 );

  //sequential addition
  if(sequential > 0){
	  // Arrays on the host (CPU)
	  //std::cout << "4" << std::endl;

	  /*for(int iter = 0; iter<matSize;iter++){
		  dev_a[iter] = new int [matSize];
		  dev_b[iter] = new int [matSize];
		  dev_c[iter] = new int [matSize];
		  for(int cur = 0; cur<matSize;cur++){
			  dev_a[iter][cur] = iter*cur;
			  dev_b[iter][cur] = iter*cur;
			  dev_c[iter][cur] = 0;
		  }
	  }*/
	  //std::cout << "5" << std::endl;

	  for(int x = 0; x<matSize; x++){
		  for(int y=0; y<matSize; y++){
			  dev_c[x][y] = dev_a[x][y] * dev_b[y][x];
		  }
	  }
	  //std::cout << "6" << std::endl;

  }else{

  /*
    Once we have host arrays containing data and we have allocated
    memory on the GPU, we have to transfer data from the host to the
    device. Again, notice the similarity to C's memcpy function.

    The first argument is the destination of the copy - in this case a
    pointer to memory allocated on the device. The second argument is
    the source of the copy. The third argument is the number of bytes
    we want to copy. The last argument is a constant that tells
    cudaMemcpy the direction of the transfer.
   */
	//for(int iter = 0; iter < matSize; iter++){
		//cudaMemcpy(dev_a, a[iter], matSize * sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(dev_b, b[iter], matSize * sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(dev_c, c[iter], matSize * sizeof(int), cudaMemcpyHostToDevice);
  
  /*
    FINALLY we get to run some code on the GPU. At this point, if you
    haven't looked at add.cu (in this folder), you should. The
    comments in that file explain what the add function does, so here
    let's focus on how add is being called. The first thing to notice
    is the <<<...>>>, which you should recognize as _not_ being
    standard C. This syntactic extension tells nvidia's cuda compiler
    how to parallelize the execution of the function. We'll get into
    details as the course progresses, but for we'll say that <<<N,
    1>>> is creating N _blocks_ of 1 _thread_ each. Each of these
    threads is executing add with a different data element (details of
    the indexing are in add.cu). 

    In larger programs, you will typically have many more blocks, and
    each block will have many threads. Each thread will handle a
    different piece of data, and many threads can execute at the same
    time. This is how cuda can get such large speedups.
   */

		//add<<<blocks, threads>>>(dev_a, dev_b, dev_c);
	  	  std::cout << "parallel in" << std::endl;
	  	  mult<<<blocks, threads>>>(matSize,dev_a, dev_b, dev_c);
	  	  cudaDeviceSynchronize();
	  	  std::cout << "parallel out" << std::endl;

  /*
    Unfortunately, the GPU is to some extent a black box. In order to
    print the results of our call to add, we have to transfer the data
    back to the host. We do that with a call to cudaMemcpy, which is
    just like the cudaMemcpy calls above, except that the direction of
    the transfer (given by the last argument) is reversed. In a real
    program we would want to check the error code returned by this
    function.
  */
		//cudaMemcpy(c[iter], dev_c, matSize * sizeof(int), cudaMemcpyDeviceToHost);
	//}
  }
  /*
    This is the other end of the timing process. We record an event,
    synchronize on it, and then figure out the difference in time
    between the start and the stop.

    We have to call cudaEventSynchronize before we can safely _read_
    the value of the stop event. This is because the GPU may not have
    actually written to the event until all other work has finished.
   */
  cudaEventRecord( end, 0 );
  cudaEventSynchronize( end );

  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, end );

  /*
    Let's check that the results are what we expect.
   */
  for (int i = 0; i < matSize; ++i) {
	  for(int j = 0; j < matSize; j++){
		  if (dev_c[i][j] != dev_a[i][j] * dev_b[j][i]) {
			  std::cerr << "Oh no! Something went wrong. :(" << std::endl;
			  std::cout << "Your program took: " << elapsedTime << " ms." << std::endl;
			  std::cout << "Values at error location were: a: " << dev_a[i][j] << " b: " << dev_b[j][i]
			            << " c: " << dev_c[i][j] << " i: " << i << " j: " << j << std::endl;

			  // clean up events - we should check for error codes here.
			  cudaEventDestroy( start );
			  cudaEventDestroy( end );

			  // clean up device pointers - just like free in C. We don't have
			  // to check error codes for this one.
			  cudaFree(dev_a);
			  cudaFree(dev_b);
			  cudaFree(dev_c);
			  exit(1);
		  }
	  }
  }

  /*
    Let's let the user know that everything is ok and then display
    some information about the times we recorded above.
   */
  std::cout << "Yay! Your program's results are correct." << std::endl;
  std::cout << "Your program took: " << elapsedTime << " ms." << std::endl;
  
  // Cleanup in the event of success.
  cudaEventDestroy( start );
  cudaEventDestroy( end );

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

}
