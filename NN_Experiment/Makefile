BRUTE=NN_Brute/
KDTREE=NN_KDtree/
DEBUG=-g -G -DDEBUG
HOME = /home/aadil/

all: nn_aadil CUDA_KDtree

# Brute Force rules
nn_aadil : $(BRUTE)nn_cuda_aadil.cu
	nvcc $(BRUTE)nn_cuda_aadil.cu -o nn_aadil -L$(HOME)ANN/lib/ -lANN -I$(HOME)ANN/include/

debug : $(BRUTE)nn_cuda_aadil.cu
	nvcc $(DEBUG) $(BRUTE)nn_cuda_aadil.cu -o debug -L$(HOME)ANN/lib/ -lANN -I$(HOME)ANN/include/

nn : nn_cuda.cu
	$(CC) -cuda nn_cuda.cu
	$(CC) -o nn nn_cuda.cu


# KDtree Rules
CUDA_KDtree: KDtree.o CUDA_KDtree.o main.o
	nvcc -O2 -o CUDA_KDtree main.o KDtree.o CUDA_KDtree.o -L$(HOME)ANN/lib/ -lANN

main.o: $(KDTREE)main.cpp
	nvcc -O2 -c $(KDTREE)main.cpp -I$(HOME)ANN/include/

KDtree.o: $(KDTREE)KDtree.cpp
	nvcc -O2 -c $(KDTREE)KDtree.cpp

CUDA_KDtree.o: $(KDTREE)CUDA_KDtree.cu
	nvcc -O2 -c $(KDTREE)CUDA_KDtree.cu

# Cleanup
clean :
	rm -rf *.o *.pyc nn nn_aadil debug CUDA_KDtree

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

plot_results: plot_results.p
	gnuplot -persist plot_results.p

#data :
#	mkdir data
#	./gen_dataset.sh
