/*
 * Author: Aadil Ahamed
 * Created: 4/25/16
 * nearest_neighbor.h: Header file for Nearest Neighbor Abstract Class.
 */

#ifndef NEAREST_NEIGHBOR_H_
#define NEAREST_NEIGHBOR_H_

#include "point.h"
#include <chrono>
#include <string>

template <typename T>
class NearestNeighbor
{
  public:

    //NearestNeighbor(){}
    NearestNeighbor(string name) : name(name) {}
    ~NearestNeighbor(){}

    /*
     * nns: nearest neighbor search
     * @param data reference data set
     * @param queries query set
     * @param result result[i] contains the index of the nearest neighbor of queries[i]
     *               Therefore nearest neighbor of query[i] is data[result[i]]
     */
    virtual void nns(vector<Point<T>> &data, vector<Point<T>> &queries, vector<int> &result) = 0;
    
    chrono::milliseconds get_search_time()
    {
      return search_time;
    }

    chrono::milliseconds get_create_time()
    {
      return create_time;
    }

    string get_name()
    {
      return name;
    }

  protected:
    string name;
    chrono::milliseconds search_time;
    chrono::milliseconds create_time;
};

#endif // NEAREST_NEIGHBOR_H_
