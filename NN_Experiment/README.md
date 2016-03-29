Nearest Neighbor Experiment
===========================

Sources:
--------
GPU KDTree Nearest Neighbor Search implementation (./NN_KDtree) was obtained from [here](http://nghiaho.com/?p=437)


Building:
--------
To build, run the command:  
```
make all
```  

Executing:
----------
To execute run the command:  
```
python driver.py
```  

Output:
-------
Output files are stored in output/  
There are 2 output files: brute_out.txt and kd_out.txt  
Each file contains 2 columns. The first column represents the number of records in both the data set and query set.  
The second column represents the running time to perform nearest neighbor search for the corresponding query/data set. 

For more information check out the [wiki](https://github.com/aahamed/Experiments/wiki/Nearest-Neighbor-Search)
