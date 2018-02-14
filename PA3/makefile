# Point to includes of local directories
CC=/usr/local/cuda-9.0/bin/nvcc
INCLUDES=-I../include
O_FILES=hellocuda.o add.o

all: $(O_FILES)
	$(CC) -o PA3 $(O_FILES)

hellocuda.o: ../src/hellocuda.cu
	$(CC) -c ../src/hellocuda.cu -o hellocuda.o $(INCLUDES)

add.o: ../src/add.cu
	$(CC) -c ../src/add.cu -o add.o $(INCLUDES)

clean:
	rm -f *.o
	rm -f *~
