
#include "Fitness.h"
#include <unistd.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__device__ bool Debug = true;

struct Coord {
public:
	int x;
	int y;
};


__device__ Coord* GetRegionTiles(int startX,int startY,int mapSize,int **map){
	if(Debug && threadIdx.x == 0){
		printf("Getting Region Tiles\n");
	}
	Coord *tiles;
	tiles = new Coord[mapSize*mapSize];
	int** mapFlags;
	int tileType = map[startX][startY];
	//init mapFlags
	mapFlags = new int*[mapSize];
	for(int flags = 0; flags < mapSize; flags++){
		mapFlags[flags] = new int[mapSize];
		for(int init = 0; init < mapSize; init++){
			mapFlags[flags][init] = 0;
		}
	}

	tiles[0].x = startX;
	tiles[0].y = startY;
	mapFlags [startX][startY] = 1;

	int iter = 0;
	int cursor = 0;
	if(Debug && threadIdx.x == 0){
		printf("%d, %d; ",tiles[iter].x,tiles[iter].y);
	}
	while (tiles[cursor].x != -1 && cursor < mapSize*mapSize) {

		for (int x = tiles[cursor].x - 1; x <= tiles[cursor].x + 1; x++) {
			for (int y = tiles[cursor].y - 1; y <= tiles[cursor].y + 1; y++) {
				if ((x >= 0  && x < mapSize && y >= 0  && y < mapSize) && (y == tiles[cursor].y || x == tiles[cursor].x)) {
					if (mapFlags[x][y] == 0 && map[x][y] == tileType) {
						mapFlags[x][y] = 1;
						iter++;
						tiles[iter].x = x;
						tiles[iter].y = y;
						if(Debug && threadIdx.x == 0){
							printf("%d, %d; ",tiles[iter].x,tiles[iter].y);
						}
					}
				}
			}
		}
		if(iter == cursor && iter < mapSize*mapSize){
			iter++;
			tiles[iter].x = -1;
			tiles[iter].y = -1;
			cursor++;
		}
		else{
			cursor++;
		}
	}

	for(int i = 0; i < mapSize; i++){
		delete(mapFlags[i]);
	}
	delete(mapFlags);

	return tiles;
}

__device__ Coord** GetRegions(int **map, int mapSize, int startX, int startY, int tileType){
	if(Debug && threadIdx.x == 0){
		printf("Getting Regions\n");
	}
	Coord** regions;
	int** mapFlags;

	//init regions
	regions = new Coord*[mapSize*mapSize];
	//init mapFlags
	mapFlags = new int*[mapSize];
	for(int flags = 0; flags < mapSize; flags++){
		mapFlags[flags] = new int[mapSize];
		for(int init = 0; init < mapSize; init++){
			mapFlags[flags][init] = 0;
		}
	}

	int cursor = 0;
	int iter = 0;
	int roomNum = 0;
	for (int row = 0; row < mapSize; row ++) {
		for (int col = 0; col < mapSize; col ++) {
			if (mapFlags[row][col] == 0 && map[row][col] == tileType) {
				Coord *newRegion = GetRegionTiles(row,col,mapSize,map);
				roomNum = 0;
				while(newRegion[roomNum].x != -1){
					roomNum++;
				}
				regions[cursor] = new Coord[roomNum];
				for(int i = 0; i < roomNum; i++){
					regions[cursor][i].x = newRegion[i].x;
					regions[cursor][i].y = newRegion[i].y;
				}
				cursor++;
				iter = 0;
				while(newRegion[iter].x != -1 && iter < mapSize*mapSize){
					mapFlags[newRegion[iter].x][newRegion[iter].y] = 1;
					if(Debug && threadIdx.x == 0){
						printf("%d, %d; ",newRegion[iter].x,newRegion[iter].y);
					}
					iter++;
				}
				if(Debug && threadIdx.x == 0){
					printf("\n");
				}
				delete(newRegion);
			}
		}
	}
	if(cursor < mapSize*mapSize){
		Coord* newRegion = new Coord[1];
		regions[cursor] = newRegion;
		regions[cursor][0].x = -1;
		regions[cursor][0].y = -1;
		delete(newRegion);
	}
	for(int i = 0; i < mapSize; i++){
		delete(mapFlags[i]);
	}
	delete(mapFlags);
	return regions;
}

__device__ void MakePassage(int **map, Coord tileA, Coord tileB){
	if(Debug && threadIdx.x == 0){
		printf("Making Passage\n");
	}
	int cursor = 0;
	int target = 0;
	int prevX = 0;
	cursor = tileA.x;
	target = tileB.x;

	if ((tileA.x - tileB.x) * (tileA.x - tileB.x) > 0) {
		if (tileA.x > tileB.x) {
			while (cursor > target) {
				cursor--;
				map [cursor] [tileA.y] = 0;
			}
		} else {
			while (cursor < target) {
				cursor++;
				map [cursor] [tileA.y] = 0;
			}
		}
	}
	prevX = cursor;
	if ((tileA.y - tileB.y) * (tileA.y - tileB.y) > 0) {
		if (tileA.y > tileB.y) {
			cursor = tileA.y;
			target = tileB.y;
			while (cursor > target) {
				cursor--;
				map [prevX] [cursor] = 0;
			}
		} else {
			cursor = tileA.y;
			target = tileB.y;
			while (cursor < target) {
				cursor++;
				map [prevX] [cursor] = 0;
			}
		}
	}
}

__device__ void ConnectClosestRooms(Coord **rooms, int **map, int mapSize){
	if(Debug && threadIdx.x == 0){
		printf("Connect Closest Rooms\n");
	}
	int roomNum = 0;
	bool **connected;
	bool *accessibleToStart;

	if(Debug && threadIdx.x == 0){
		printf("Finding Room total\n");
	}
	while(rooms[roomNum][0].x != -1 && roomNum < mapSize*mapSize){
		roomNum++;
	}

	if(Debug && threadIdx.x == 0){
		printf("setting connected tables\n");
	}
	connected = new bool*[roomNum];
	accessibleToStart = new bool[roomNum];
	for(int row = 0; row < roomNum; row++){
		connected[row] = new bool[roomNum];
		accessibleToStart[row] = false;
		for(int col = 0; col < roomNum; col++){
			connected[row][col] = false;
			if(row == col){
				connected[row][col] = true;
			}
		}
	}
	accessibleToStart[0] = true;
	//while there are disconnected rooms
	int access = 1;
	Coord bestTileA;
	Coord bestTileB;
	int bestRoom1;
	int bestRoom2;
	bool possibleConnection = false;
	int bestDistance = 0;

	if(Debug && threadIdx.x == 0){
		printf("Connecting Rooms\n");
	}
	while(access < roomNum){
		possibleConnection = false;
		bestDistance = 0;
		bestRoom1 = 0;
		bestRoom2 = 0;
		//find two nearest rooms
		int tileA = 0;
		int tileB = 0;
		for(int room = 0; room < roomNum; room++){
			for(int room2 = 0; room2 < roomNum; room2++){
				if(!connected[room][room2] && !connected[room2][room]){
					tileA = 0;
					while(rooms[room][tileA].x != -1 && tileA < mapSize*mapSize){
						tileB = 0;
						while(rooms[room2][tileB].x != -1 && tileB < mapSize*mapSize){
							int distance = (rooms[room][tileA].x - rooms[room2][tileB].x) * (rooms[room][tileA].x - rooms[room2][tileB].x);
							distance = distance + (rooms[room][tileA].y - rooms[room2][tileB].y) * (rooms[room][tileA].y - rooms[room2][tileB].y);

							if(Debug && threadIdx.x == 0){
								printf("Rooms: %d, %d;\n",room,room2);
								printf("Tile: %d, %d;\n",rooms[room][tileA].x,rooms[room][tileA].y);
								printf("Tile: %d, %d;\n",rooms[room2][tileB].x,rooms[room2][tileB].y);
							}
							if(distance < bestDistance || !possibleConnection){
								bestDistance = distance;
								possibleConnection = true;
								bestTileA.x = rooms[room][tileA].x;
								bestTileA.y = rooms[room][tileA].y;
								bestTileB.x = rooms[room2][tileB].x;
								bestTileB.y = rooms[room2][tileB].y;
								bestRoom1 = room;
								bestRoom2 = room2;
							}
							tileB++;
						}
						tileA++;
					}
				}
			}
		}
		//connect those rooms
		if(Debug && threadIdx.x == 0){
			//printf("%d, %d;\n",bestTileA.x,bestTileA.y);
			//printf("%d, %d;\n",bestTileB.x,bestTileB.y);
		}
		MakePassage(map,bestTileA,bestTileB);
		connected[bestRoom1][bestRoom2] = true;
		connected[bestRoom2][bestRoom1] = true;
		//check if accessible to start
		if((accessibleToStart[bestRoom1] && !accessibleToStart[bestRoom2])){
			accessibleToStart[bestRoom2] = true;
			access++;
			for(int iter = 0; iter < roomNum; iter++){
				if(connected[bestRoom2][iter] && !accessibleToStart[iter]){
					accessibleToStart[iter] = true;
					access++;
				}
			}
		}
		else if((!accessibleToStart[bestRoom1] && accessibleToStart[bestRoom2])){
			accessibleToStart[bestRoom1] = true;
			access++;
			for(int iter = 0; iter < roomNum; iter++){
				if(connected[bestRoom1][iter] && !accessibleToStart[iter]){
					accessibleToStart[iter] = true;
					access++;
				}
			}
		}
	}
	delete(accessibleToStart);
	for(int i = 0; i < roomNum; i++){
		delete(connected[i]);
		delete(rooms[i]);
	}
	delete(rooms);
	delete(connected);
}

__device__ int GetNeighbors(int **map, int x, int y, int mapSize){
	int neighbors = 0;
	int row = x-1;
	int col = y-1;
	int rowMax = x+1;
	int colMax = y+1;

	//check for out of bounds
	if(row < 0){
		row = 0;
	}
	if (rowMax >= mapSize){
		rowMax = mapSize-1;
	}
	if (colMax >= mapSize){
		colMax = mapSize-1;
	}
	//get neighbors
	for(; row < rowMax; row++){
		for(col = y-1; col < colMax; col++){
			if(col < 0){
				col = 0;
			}
			if(!(row == x && col == y) && map[row][col] == 1){
				neighbors++;
			}
		}
	}

	return neighbors;
}

__device__ void RunCA(int **map, int mapSize, float* rules, unsigned int seed){
	int **CAmap;
	int maxCA = 50;

	CAmap = new int*[mapSize];
	for(int iter = 0; iter < mapSize; iter++){
		CAmap[iter] = new int[mapSize];
	}
	curandState_t state;

	  /* we have to initialize the state */
	  curand_init(seed, /* the seed controls the sequence of random values that are produced */
	              0, /* the sequence number is only important with multiple cores */
	              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	              &state);

	for(int caIter = 0; caIter < maxCA; caIter++){
		for(int x = 0; x < mapSize; x++){
			for(int y = 0; y < mapSize; y++){
				//rules stuff here
				float rand = curand(&state)%127;
				if(map[x][y] < 1 && rules[GetNeighbors(map,x,y,mapSize)] > rand){
					CAmap[x][y] = 1;
				}
				else if(map[x][y] > 0 && rules[GetNeighbors(map,x,y,mapSize)+9] > rand){
					CAmap[x][y] = 0;
				}
			}
		}
		//set map equal to CAmap
		for(int x = 0; x < mapSize; x++){
			for(int y = 0; y < mapSize; y++){
				map[x][y] = CAmap[x][y];
				}
		}
	}
	for(int i = 0; i < mapSize; i++){
		delete(CAmap[i]);
	}
	delete(CAmap);
}

/*
  This is the function that each thread will execute on the GPU. The
  fact that it executes on the device is indicated by the __global__
  modifier in front of the return type of the function. After that,
  the signature of the function isn't special - in particular, the
  pointers we pass in should point to memory on the device, but this
  is not indicated by the function's signature.
 */
__global__ void GetFitnesses(float **population, float *fitnesses, int popSize,int chromSize,int mapSize, unsigned int seed) {
	int chrom = (threadIdx.x)%popSize;
	int** map;
	//printf("%d, ",chrom);
	//init map
	map = new int*[mapSize];
	for(int iter = 0; iter < mapSize; iter++){
		map[iter] = new int[mapSize];
		for(int curs = 0; curs < mapSize; curs++){
			map[iter][curs] = 0;
		}
	}

	//build maze pattern
	RunCA(map,mapSize,population[chrom],seed);
	//clear start and end tiles
	map[0][0] = 0;
	map[mapSize-1][mapSize-1] = 0;
	if(threadIdx.x == 0 && Debug){
		for(int x = 0; x < mapSize; x++){
			for(int y = 0; y < mapSize; y++){
				printf("%d",map[x][y]);
			}
			printf("\n");
		}
	}
	//find disconnected rooms
	Coord **rooms = GetRegions(map,mapSize,0,0,0);

	//connect disconnected rooms
	if(threadIdx.x == 0){
	ConnectClosestRooms(rooms,map,mapSize);
	}
	/*int roomNum = 0;
	while(rooms[roomNum][0].x != -1){
		roomNum++;
	}
	for(int temp = 0; temp < roomNum; temp++){
		delete(rooms[temp]);
	}
	delete(rooms);*/

	//Check fitness of maze

	for(int i = 0; i < mapSize; i++){
		delete (map[i]);
	}
	delete(map);
}


__global__ void mult(int size,int** a, int** b, int** c, int** d, int** e) {
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
		c[x][y] = (a[x][y] * b[y][x]) + (d[x][y] * e[y][x]);
	}
}
