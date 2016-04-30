GPP=g++
STD=-std=c++11
DEBUG=-DDEBUG
HOME = /home/aadil/

test_point: test_point.cpp point.h
	$(GPP) $(STD) test_point.cpp -o test_point


test_brute: brute.cpp
	$(GPP) $(STD) $(DEBUG) brute.cpp -o test_brute 

test_ann: ann_imp.cpp
	$(GPP) $(STD) $(DEBUG) ann_imp.cpp -o test_ann -L$(HOME)ANN/lib/ -lANN -I$(HOME)ANN/include/

test_environment: ann_imp.cpp
	$(GPP) $(STD) $(DEBUG) test_environment.cpp -o test_environment -L$(HOME)ANN/lib/ -lANN -I$(HOME)ANN/include/

clean:
	rm -rf test_point test_brute test_ann test_environment
