GPP=g++
NVCC=nvcc
STD=-std=c++11
DEBUG=-DDEBUG -g -G
HOME = /home/aadil/

test_point: test_point.cpp point.h
	$(NVCC) $(STD) test_point.cpp -o test_point

#point: point.cpp
#	$(NVCC) $(STD) -c point.cpp -o point.o

test_brute: brute.cpp
	$(NVCC) $(STD) $(DEBUG) brute.cpp -o test_brute 

brute: brute.cpp
	$(NVCC) $(STD) -c brute.cpp -o brute.o

test_ann: ann_imp.cpp
	$(NVCC) $(STD) $(DEBUG) ann_imp.cpp -o test_ann -L$(HOME)ANN/lib/ -lANN -I$(HOME)ANN/include/

ann: ann_imp.cpp
	$(NVCC) $(STD) -c ann_imp.cpp -o ann.o -L$(HOME)ANN/lib/ -lANN -I$(HOME)ANN/include/

brute_gpu: brute_gpu.cu
	$(NVCC) $(STD) -c brute_gpu.cu -o brute_gpu.o

test_brute_gpu: brute brute_gpu.cu
	$(NVCC) $(STD) $(DEBUG) brute.o brute_gpu.cu -o brute_gpu

test_environment: brute ann brute_gpu test_environment.cpp
	$(NVCC) $(STD) $(DEBUG) *.o test_environment.cpp -o test_environment -L$(HOME)ANN/lib/ -lANN -I$(HOME)ANN/include/

clean:
	rm -rf test_point test_brute test_ann test_environment brute_gpu *.o
