nvcc -c main_1.cu -arch=sm_20
g++ -o main_gpu_1 main_1.o `OcelotConfig -l`
./main_gpu_1
