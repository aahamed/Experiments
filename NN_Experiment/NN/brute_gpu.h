/*
 * Author: Aadil Ahamed
 * Created: 5/16/16
 * brute.h: Header file for brute force implementation on GPU of Nearest Neighbor.
 */

#ifndef BRUTE_GPU_H 
#define BRUTE_GPU_H

#include "brute.h"
#include "cuda.h"
#include <assert.h>

#define square(x) ((x)*(x))

template <typename T>
class BruteGPU: public Brute<T>
{
  public:
    BruteGPU(string name = "BruteGPU" ) : Brute<T>(name){}
    ~BruteGPU(){}
    /*
     * nns: nearest neighbor search
     * @param data reference data set
     * @param queries query set
     * @param result result[i] contains the index of the nearest neighbor of queries[i]
     *               Therefore nearest neighbor of query[i] is data[result[i]]
     */
    virtual void nns(vector<Point<T>> &data, vector<Point<T>> &queries, vector<int> &result);
};

/*
template <typename T>
__global__ void nn(vector<Point<T>> &data, vector<Point<T>> &queries, vector<int> &result);

template <typename T>
__device__ float serial_distance(Point<T> &p, Point<T> &q);
*/


#endif
