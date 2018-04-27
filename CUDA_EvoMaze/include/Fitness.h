/*
  This header demonstrates how we build cuda programs spanning
  multiple files. 
 */

#ifndef FITNESS_H_
#define FITNESS_H_

// This is the number of elements we want to process.
//#define N 1024

struct Coord;
// This is the declaration of the function that will execute on the GPU.
__device__ Coord* GetRegionTiles(int ,int ,int ,int **);

__device__ Coord** GetRegions(int **, int , int , int , int );

__global__ void GetFitnesses(float **, float *,int,int,int, unsigned int);

__global__ void mult(int, int**, int**, int**, int**, int**);

#endif // ADD_H_
