/*
 * Author: Aadil Ahamed
 * Created: 4/28/16
 * ann_imp.h: Header file for ANN Library implentation of Nearest Neighbor Class.
 */

#ifndef ANN_IMP_H_
#define ANN_IMP_H_

#include "nearest_neighbor.h"
#include <ANN/ANN.h>
#include <assert.h>

template <typename T>
class AnnImp: public NearestNeighbor<T> 
{
  public:
    AnnImp(string name = "ANN" ) : NearestNeighbor<T>(name) {}
    ~AnnImp(){}

    /*
     * nns: nearest neighbor search
     * @param data reference data set
     * @param queries query set
     * @param result result[i] contains the index of the nearest neighbor of queries[i]
     *               Therefore nearest neighbor of query[i] is data[result[i]]
     */
    void nns(vector<Point<T>> &data, vector<Point<T>> &queries, vector<int> &result);
  private:
    void ann_search( vector<Point<T>> &data, vector<Point<T>> &queries, vector<int> &result, ANNkd_tree *kd_tree, int k=1);
};

#endif // ANN_IMP_H_
