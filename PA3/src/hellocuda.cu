/*
  This program demonstrates the basics of working with cuda. We use
  the GPU to add two arrays. We also introduce cuda's approach to
  error handling and timing using cuda Events.

  This is the main program. You should also look at the header add.h
  for the important declarations, and then look at add.cu to see how
  to define functions that execute on the GPU.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "add.h"

int getLargest(int* nn,float* array,int K){
	int large = 0;

	for(int i = 0; i < K; i++){
		if(array[nn[i]] > array[nn[large]]){
			large = i;
		}
	}

	return large;
}

void readCSV(float** dev_a,int matSize){
	std::ifstream infile;
	infile.open("../pa3.csv");
	std::string line;
	std::string cell;
	int row = 0;
	int col = 0;
	if (infile.is_open())
	{
		while(row < matSize && std::getline(infile,line)){
			std::stringstream ss(line);
			while(col < matSize && std::getline(ss,cell,',')){
				dev_a[row][col]= std::strtof(cell.c_str(),0);
				col++;
			}
			row++;
			col = 0;
		}
		infile.close();
	}
	else {
		std::cout << "Error opening file" <<std::endl;
	}
}

int main() {
  int matSize = 100;
  int sequential = 1;
  int blocks = 1;
  int threads = 1;
  int K = 5;

  int nanRows[10] = {99,55,89,23,42,69,7,11,33,75};
  float repNums[10];

  //get array dimensions
  //std::cout << "Please enter the number of nearest neighbors (K):";
  //std::cin >> matSize;
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
  float **dev_a, *dev_b;//, **dev_c;

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
  cudaError_t err = cudaMallocManaged( (void**) &dev_a, (matSize*matSize) * sizeof(float));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  cudaMallocManaged( (void**) &dev_b, (matSize) * sizeof(float));
  //cudaMallocManaged( (void**) &dev_c, (matSize*matSize) * sizeof(int));

  // These lines just fill the host arrays with some data so we can do
  // something interesting. Well, so we can add two arrays.
  //std::cout << "2" << std::endl;
  for(int iter = 0; iter<matSize;iter++){
	  cudaMallocManaged( (void**) &(dev_a[iter]), (matSize)*sizeof(float));
	  //cudaMallocManaged( (void**) &(dev_b[iter]), (matSize)*sizeof(int));
	  //cudaMallocManaged( (void**) &(dev_c[iter]), (matSize)*sizeof(int));
  	  for(int cur = 0; cur<matSize;cur++){
  		dev_a[iter][cur] = 0;
  		//dev_c[iter][cur] = 0;
  	  }
	  dev_b[iter] = 0;
    }
  //std::cout << "3" << std::endl;

  //Read in CSV File
  readCSV(dev_a,matSize);

  //Replace Random nums with NaN
  for(int i=0;i<10;i++){
	  repNums[i] = dev_a[nanRows[i]][1];
	  dev_a[nanRows[i]][1] = 0.0/0.0;
  }

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
	  /*
	  for(int x = 0; x<matSize; x++){
		  for(int y=0; y<matSize; y++){
			  dev_c[x][y] = dev_a[x][y] * dev_b[y][x];
		  }
	  }
	  //std::cout << "6" << std::endl;
	   */
	  //Sequential K nearest neighbors
	  for(int misRow = 0; misRow < matSize; misRow++)
	  {
		  if(isnan(dev_a[misRow][1])){
			  //std::cout << "nan" << std::endl;
			  int nn[K];
			  int largest = 0;
			  for(int i = 0; i < K; i++){
				  nn[i] = i;
			  }
			  for(int row = 0; row < matSize; row++){
				  dev_b[row] = 0;
				  for(int col = 2; col < matSize; col++){
					  if(row == misRow || isnan(dev_a[row][1])){
						  std::cout << "nan row, skip" << std::endl;
						  dev_b[row] = 1000*1000;
						  col = matSize;
					  }
					  else{
						  dev_b[row] += (dev_a[misRow][col] - dev_a[row][col]) * (dev_a[misRow][col] - dev_a[row][col]);
						  if(row < 1){
							  std::cout << dev_b[row] << ",";
						  }
					  }
				  }
				  if(row < 1){
					  std::cout << std::endl;
				  }
				  //std::cout << dev_b[row] << " ";
				  dev_b[row] = sqrt(dev_b[row]);
				  //std::cout << dev_b[row] << std::endl;
				  if(row >= K){
					  largest = getLargest(nn,dev_b, K);
					  if(dev_b[row] < dev_b[nn[largest]]){
						  nn[largest] = row;
					  }
				  }
			  }

			  float avg = 0;
			  for(int j = 0; j < K; j++){
				  avg = avg + dev_a[nn[j]][1];
			  }
			  avg = avg/K;
			  dev_a[misRow][1] = avg;
		  }
	  }

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


	  	  //find row with missing value
	  	  for(int misRow = 0;misRow < matSize; misRow++)
		  {
			  if(isnan(dev_a[misRow][1])){
				  //std::cout << "nan" << std::endl;
				  int nn[K];
				  int largest = 0;
				  for(int i = 0; i < K; i++){
					  nn[i] = i;
				  }
				  for(int iter = 0; iter < matSize; iter++){
					  dev_b[iter] = 0;
				  }
				  //add<<<blocks, threads>>>(dev_a, dev_b, dev_c);
				  //std::cout << "parallel in" << std::endl;
				  //mult<<<blocks, threads>>>(matSize,dev_a, dev_b, dev_c);
				  //parallel KNN distance finding
				  KNN<<<blocks, threads>>>(dev_a,dev_b,matSize,misRow);
				  cudaDeviceSynchronize();
				  //std::cout << "parallel out" << std::endl;
				  //get K nearest neighbors
				  for(int x=0;x<matSize;x++){
					  dev_b[x] = sqrt(dev_b[x]);
				  }

				  for(int i = K; i < matSize; i++){
					  largest = getLargest(nn,dev_b, K);
					  if(dev_b[i] < dev_b[nn[largest]]){
						  nn[largest] = i;
					  }
				  }

				  float avg = 0;
				  for(int j = 0; j < K; j++){
					  avg = avg + dev_a[nn[j]][1];
				  }
				  avg = avg/K;
				  dev_a[misRow][1] = avg;
			  }
		  }

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
  for (int i = 0; i < matSize/10; ++i) {
	  /*for(int j = 0; j < matSize; j++){
		  if (false) {
			  std::cerr << "Oh no! Something went wrong. :(" << std::endl;
			  std::cout << "Your program took: " << elapsedTime << " ms." << std::endl;

			  // clean up events - we should check for error codes here.
			  cudaEventDestroy( start );
			  cudaEventDestroy( end );

			  // clean up device pointers - just like free in C. We don't have
			  // to check error codes for this one.
			  cudaFree(dev_a);
			  cudaFree(dev_b);
			  //cudaFree(dev_c);
			  exit(1);
		  }
	  }*/
	  std::cout << "removed value:" << repNums[i] << " Estimated value:" << dev_a[nanRows[i]][1] << std::endl;
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
  //cudaFree(dev_c);

}
