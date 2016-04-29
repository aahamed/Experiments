/*
 * Author: Aadil Ahamed
 * Created: 4/25/16
 * nearest_neighbor.h: Header file for Nearest Neighbor Abstract Class.
 */

#ifndef NEAREST_NEIGHBOR_H_
#define NEAREST_NEIGHBOR_H_

#include "point.h"

template <typename T>
class NearestNeighbor
{
  public:
    virtual void nns(vector<Point<T>> &data, vector<Point<T>> &queries, vector<Point<T>> &result) = 0;
};

#endif // NEAREST_NEIGHBOR_H_
