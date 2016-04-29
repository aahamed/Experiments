/*
 * Author: Aadil Ahamed
 * Created: 4/28/16
 * nearest_neighbor.h: Header file for Nearest Neighbor Abstract Class.
 */

#ifndef ANN_IMP_H_
#define ANN_IMP_H_

#include "nearest_neighbor.h"
#include <ANN/ANN.h>

template <typename T>
class AnnImp: public NearestNeighbor<T> 
{
  public:
    void nns(vector<Point<T>> &data, vector<Point<T>> &queries, vector<Point<T>> &result);
  private:
    void ann_search( vector<Point<T>> &data, vector<Point<T>> &queries, vector<Point<T>> &result, ANNkd_tree *kd_tree, int k=1);
};

#endif // ANN_IMP_H_
