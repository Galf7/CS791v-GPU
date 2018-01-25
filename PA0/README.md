# PA0: Add Two Arrays
You can execute PA0 in your local machine or Cubix (a GPU cluster). If you would like to use Cubix, you need to ask Zachary to create an account for you. I have listed how to execute PA0 in your local machine and Cubix in this readme file.

# Local machine
## Dependencies, Building, and Running

### Dependency Instructions
First, you need to test if the graphic card driver is installed by using this command:
```
nvidia-smi
```

If you cannot communicate with your driver by using this command, then it means you have not installed the driver. To fix this, please follow these [instructions](https://askubuntu.com/questions/851069/latest-nvidia-driver-on-ubuntu-16-04). After this step, try the 

```
nvidia-smi
```

command again to test if you can communicate with the graphic card.

PA0 requires CUDA toolkit and we will use CUDA 9.0 in this assignment. Details about the CUDA introduction and dependencies are available at [CUDA 9.0](https://developer.nvidia.com/cuda-toolkit/whatsnew). You can download the CUDA 9.0 toolkit at [CUDA 9.0 download](https://developer.nvidia.com/cuda-90-download-archive). 

After the installation, you need to follow these commands to setup envriontmental parameters:
```Environmental Parameters Setup
export CUDA_HOME=/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
PATH=${CUDA_HOME}/bin:${PATH}
export PATH
```
<!-- You may need to repeat the "Environmental Parameters Setup" every time when you open a new terminal. Because you bashrc does not have PATH parameter. If this happens, try the following command:
```
echo "export PATH=/usr/local/cuda-9.0/bin:$PATH" >> ~/.bashrc
```
 -->

Please follow the instructions and install the toolkit and test it by executing PA0.

### Building and Running
To build this project, you need to use the provided Makefile which is used as usual.

Running the make in a separate directory will allow easy cleanup of the build data, and an easy way to prevent unnecessary data to be added to the git repository.  

#### Makefile Instructions 
The makefile works as expected and must be updated with new files added in.

```bash
mkdir build
cd build
cp ../makefile .
make
./PA0
```

If you want to clean the build folder, then execute the following command:
```
make clean
```

# Cubix
The dependencies have been installed in Cubix and you need to follow these commands to setup envriontmental parameters:
```Environmental Parameters Setup
export CUDA_HOME=/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
PATH=${CUDA_HOME}/bin:${PATH}
export PATH
```


Please follow the instructions and install the toolkit and test it by executing PA0.

### Building and Running
To build this project, you need to use the provided Makefile which is used as usual.

Running the make in a separate directory will allow easy cleanup of the build data, and an easy way to prevent unnecessary data to be added to the git repository.  

#### Makefile Instructions 
The makefile works as expected and must be updated with new files added in.

```bash
mkdir build
cd build
cp ../makefile .
make
srun --gres=gpu:1 ./PA0
```

If you execute any GPU program in Cubix, you need to use [slurm](https://slurm.schedmd.com/quickstart.html). Slurm is a workload manager and manages all the execution requests in queues. ```--gres=gpu:1``` means one GPU is used to execute the program.
