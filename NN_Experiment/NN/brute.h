/*
 * Author: Aadil Ahamed
 * Created: 4/25/16
 * brute.h: Header file for brute force implementation of Nearest Neighbor.
 */

#include "nearest_neighbor.h"

template <typename T>
class Brute: public NearestNeighbor<T>
{
  public:
    void nns(vector<Point<T>> &data, vector<Point<T>> &queries, vector<Point<T>> &result);

  private:
    float serial_distance(Point<T> &p, Point<T> &q);
};

