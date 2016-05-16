/*
 * Author: Aadil Ahamed
 * Created: 4/25/16
 * brute.h: Header file for brute force implementation of Nearest Neighbor.
 */

#ifndef BRUTE_H 
#define BRUTE_H

#include "nearest_neighbor.h"
#include <assert.h>

#define square(x) ((x)*(x))

template <typename T>
class Brute: public NearestNeighbor<T>
{
  public:
    Brute(string name = "Brute" ) : NearestNeighbor<T>(name){}
    ~Brute(){}
    /*
     * nns: nearest neighbor search
     * @param data reference data set
     * @param queries query set
     * @param result result[i] contains the index of the nearest neighbor of queries[i]
     *               Therefore nearest neighbor of query[i] is data[result[i]]
     */
    virtual void nns(vector<Point<T>> &data, vector<Point<T>> &queries, vector<int> &result);

  private:
    float serial_distance(Point<T> &p, Point<T> &q);
};


#endif
