nvcc -c FFT.cu -arch=sm_20
g++ -o FFT_cu FFT.o `OcelotConfig -l`
./FFT_cu
