/*
  This program demonstrates the basics of working with cuda. We use
  the GPU to add two arrays. We also introduce cuda's approach to
  error handling and timing using cuda Events.

  This is the main program. You should also look at the header add.h
  for the important declarations, and then look at add.cu to see how
  to define functions that execute on the GPU.
 */

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <math.h>

#include "Fitness.h"

bool Debug = false;

void PopInit(float** population, int popSize,int chromSize){
	for(int pop = 0; pop < popSize; pop++){
		for(int chrom = 0; chrom < chromSize; chrom++){
			population[pop][chrom] = std::rand() % 128;
		}
	}
}

void Mutate(int *chrom,int chromSize){
	int mutateRate = 10;
	for(int iter = 0; iter < chromSize*7; iter++){
		if(mutateRate > rand()%1000){
			chrom[iter] = 1 - chrom[iter];
		}
	}
}

void Crossover(float *par1, float* par2,int chromSize, float **population, int popIter){
	int crossRate = 90;
	if (crossRate > rand()%100){
		int bin1[chromSize*7];
		int bin2[chromSize*7];

	    if(Debug){
	    	std::cout << "Convert parents to binary form" << std::endl;
	    }
		//convert chroms to binary
		int digit1 = 0;
		int digit2 = 0;
		for(int chrom = 0; chrom < chromSize; chrom++){
			digit1 = (int)(par1[chrom]);
			digit2 = (int)(par2[chrom]);
			for(int iter = chrom*7; iter < (chrom+1)*7; iter++){
				bin1[iter] = digit1%2;
				bin2[iter] = digit2%2;

				digit1 = digit1/2;
				digit2 = digit2/2;
			}
		}

	    if(Debug){
	    	std::cout << "Find differing indices" << std::endl;
	    }
		//find differing indexes
		int diffBits = 0;
		for(int chrom = 0; chrom < chromSize*7;chrom++){
			if(bin1[chrom] != bin2[chrom]){
				diffBits++;
			}
		}
		int diffIndexes[diffBits];
		int iter = 0;
		for(int chrom = 0; chrom < chromSize*7;chrom++){
			if(iter < diffBits && bin1[chrom] != bin2[chrom]){
				diffIndexes[iter] = chrom;
				iter++;
			}
		}

	    if(Debug){
	    	std::cout << "randomly swap half of the bits" << std::endl;
	    }
		//randomly choose half to swap
		std::random_shuffle(diffIndexes,diffIndexes + diffBits);
		for(int diff = 0; diff < (diffBits/2) + (diffBits%2); diff++){
			int temp = bin1[diffIndexes[diff]];
			bin1[diffIndexes[diff]] = bin2[diffIndexes[diff]];
			bin2[diffIndexes[diff]] = temp;
		}

	    if(Debug){
	    	std::cout << "Check for Mutation" << std::endl;
	    }
		//check both children for mutation
		Mutate(bin1,chromSize);
		Mutate(bin2,chromSize);

	    if(Debug){
	    	std::cout << "Convert back to float and reinsert to population" << std::endl;
	    }
		//translate children back to floats and put in pop
		int binNum1 = 0;
		int binNum2 = 0;
		for(int chrom = 0; chrom < chromSize; chrom++){
			for(int iter = chrom*7; iter < (chrom+1)*7;iter++){
				binNum1 = binNum1 + bin1[iter] * pow(2,(iter+1)%7);
				binNum2 = binNum2 + bin2[iter] * pow(2,(iter+1)%7);
			}
			population[popIter][chrom] = binNum1;
			population[popIter+1][chrom] = binNum2;
			binNum1 = 0;
			binNum2 = 0;
		}
	}
	else{
		for(int chrom = 0; chrom < chromSize; chrom++){
			population[popIter][chrom] = par1[chrom];
			population[popIter+1][chrom] = par2[chrom];
		}
	}
}

void ParentSelection(float **population, float *fitness, int popSize,int chromSize){
	int lastFit = 0;
	int curBest = 0;
	int cursor = 0;
	int bestFit = 0;
	float totFit = 0;

	int *fitList;
	fitList = new int[popSize];

    if(Debug){
    	std::cout << "initializing parents array" << std::endl;
    }

	float** parents;
	parents = new float*[popSize];
	for(int cursor = 0; cursor < popSize; cursor++){
		parents[cursor] = new float[chromSize];
	}

    if(Debug){
    	std::cout << "Finding highest fitness individual" << std::endl;
    }

	for(int i = 0; i < popSize; i++)
	{
		if (fitness [i] >= bestFit) {
			curBest = i;
			bestFit = fitness [curBest];
		}
		totFit = totFit + fitness[i];
	}
	fitList [cursor] = curBest;
	lastFit = curBest;
	bestFit = 0;
	cursor++;
	//force total fitness to be > 0 to prevent errors
	if(totFit < 1){
		totFit = 1;
	}

    if(Debug){
    	std::cout << "Sorting remaining fitness" << std::endl;
    }
	while (cursor < popSize) {
		for(int i = 0; i < popSize; i++)
		{
			if (fitness [i] >= bestFit && fitness[i] <= fitness[lastFit]) {
				if (i < lastFit) {
					curBest = i;
					bestFit = fitness [curBest];
				} else if (fitness [i] < fitness [lastFit]) {
					curBest = i;
					bestFit = fitness [curBest];
				}
			}
		}
		fitList [cursor] = curBest;
		lastFit = curBest;
		bestFit = 0;
		cursor++;
	}

	//Fitness proportional selection on top individuals to pick two
	int selected = (std::rand())%((int)totFit);
	int selectTotal = 0;
	int fit = 0;
	int iter = -1;

    if(Debug){
    	std::cout << "Fitness Proportional Selection" << std::endl;
    }

	while (selectTotal < popSize) {
		do {
			iter++;
			fit += fitness [(fitList[iter])];
		} while(fit < selected);
		//copy to parents
		for(int chrom = 0; chrom < chromSize; chrom++){
			parents[selectTotal][chrom] = population[fitList[iter]][chrom];
		}
		iter = -1;
		selected = std::rand()%(int)totFit;
		selectTotal++;
		fit = 0;
	}

    if(Debug){
    	std::cout << "Crossover Parents" << std::endl;
    }

	for(int pair = 0; pair < popSize/2; pair+=2){

		//Crossover two selected, place children back into pop
		Crossover(parents[pair], parents[pair+1], chromSize, population, pair);
		//Put parents back into population
		for(int chrom = 0; chrom < chromSize; chrom++){
			population[pair+(popSize/2)][chrom] = parents[pair][chrom];
			population[(pair+1)+(popSize/2)][chrom] = parents[(pair+1)][chrom];
		}
	}
	delete (fitList);
	for(int iter = 0; iter < popSize; iter++){
		delete(parents[iter]);
	}
	delete(parents);
}

int main() {
  int popSize;
  int chromSize;
  bool GaStop = false;
  int generation = 0;
  int genMax = 1;
  int mapSize;
  unsigned int random;

  //get array dimensions
  //std::cout << "Please enter the dimensions of the matrix (1000<=matSize<=10000):";
  //std::cin >> popSize;

  /*std::cout << "Please enter the number of blocks to be used:";
  std::cin >> blocks;
  if(blocks < 1 ){//|| blocks > 256){
  	std::cout << "invalid block number, using default of 256 (Max 65535)." << std::endl;
   	blocks = popSize;
  }*/

  /*std::cout << "Please enter the number of threads per block:";
  std::cin >> threads;
  if(threads < 1 ){//|| threads > 32){
   	std::cout << "invalid thread number, using default of 32 (Max 65535)." << std::endl;
   	threads = 32;
  }*/
  if(Debug){
	  std::cout << "allocating population and fitness arrays" << std::endl;
  }

  float **population;
  float *fitness;
  //std::cout << "0" << std::endl;

  cudaMallocManaged( (void**) &popSize, sizeof(int));
  cudaMallocManaged( (void**) &mapSize, sizeof(int));
  cudaMallocManaged( (void**) &chromSize, sizeof(int));
  cudaMallocManaged( (void**) &random, sizeof(unsigned int));

  //std::cout << "1" << std::endl;

  popSize = 4;
  chromSize = 18;
  mapSize = 30;
  random = (unsigned int)(std::rand());
  int blocks = 1;
  int threads = popSize;

  //std::cout << "2" << std::endl;
  cudaError_t err = cudaMallocManaged( (void**) &population, (popSize*chromSize) * sizeof(float));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  cudaMallocManaged( (void**) &fitness, (popSize) * sizeof(float));

  // These lines just fill the host arrays with some data so we can do
  // something interesting. Well, so we can add two arrays.
  if(Debug){
	  std::cout << "initializing population and fitness arrays" << std::endl;
  }
  for(int iter = 0; iter<popSize;iter++){
	  cudaMallocManaged( (void**) &(population[iter]), (chromSize)*sizeof(float));
	  fitness[iter] = 0;
  	  for(int cur = 0; cur<chromSize;cur++){
  		population[iter][cur] = 0;
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

	  //CUTThreads here
	  //int numGPU;
	  //cudaGetDeviceCount(&numGPU);
	  //CUTThread *thread = new CUTThread[numGPU];
	  //matStruct *matricies = new matStruct[numGPU];
	  //mult<<<blocks, threads>>>(matSize,dev_a, dev_b, dev_c, dev_d, dev_e);
	  //cudaDeviceSynchronize();

  //Randomly initialize the population
  if(Debug){
	  std::cout << "Initialize population with random values" << std::endl;
  }
  PopInit(population,popSize,chromSize);
  generation = 0;
  //Parallel Evoluitionary Cellular Automata
  while(!GaStop){
	  random = (unsigned int)(std::rand());
	  //Evaluate fitness on GPU
	  GetFitnesses<<<blocks, threads>>>(population,fitness,popSize,chromSize,mapSize,random);
	  //sync device
	  cudaDeviceSynchronize();

	  //GA Selection, Crossover, and Mutate process
	  if(Debug){
		  std::cout << "Selecting Parents for Crossover" << std::endl;
	  }
	  ParentSelection(population,fitness,popSize,chromSize);
	  //Check for stopping condition
	  if(generation < genMax){
		  //if(Debug){
			  std::cout << "Next Generation" << std::endl;
		  //}
		  generation++;
	  }
	  else{
		  std::cout << "GA Done!" << std::endl;
		  GaStop = true;
	  }
  }

  cudaEventRecord( end, 0 );
  cudaEventSynchronize( end );

  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, end );
  std::cout << "Your program took: " << elapsedTime << " ms." << std::endl;

  /*
    Let's let the user know that everything is ok and then display
    some information about the times we recorded above.
   */
  
  // Cleanup in the event of success.
  cudaEventDestroy( start );
  cudaEventDestroy( end );

  cudaFree(population);
  cudaFree(fitness);
  //cudaFree(dev_c);

}
