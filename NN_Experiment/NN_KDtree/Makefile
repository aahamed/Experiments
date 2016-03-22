HOME = /home/aadil/
CUDA_KDtree: KDtree.o CUDA_KDtree.o main.o
	nvcc -O2 -o CUDA_KDtree main.o KDtree.o CUDA_KDtree.o -L$(HOME)ANN/lib/ -lANN

main.o: main.cpp
	nvcc -O2 -c main.cpp -I$(HOME)ANN/include/

KDtree.o: KDtree.cpp
	nvcc -O2 -c KDtree.cpp

CUDA_KDtree.o: CUDA_KDtree.cu
	nvcc -O2 -c CUDA_KDtree.cu

clean:
	rm -f *.o CUDA_KDtree
