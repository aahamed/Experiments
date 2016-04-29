/*
 * Author: Aadil Ahamed
 * Created: 4/25/16
 * point.h: Header file for Point class
 */

#ifndef POINT_H_
#define POINT_H_

#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

template <typename T>
class Point
{
  public:
    Point(vector<T> &in)
    : coords(in)
    {}

    Point(const Point<T> &p)
    : coords(p.coords)
    {}

    ~Point(){}

    void print(ostream &out)
    {
      out << "(";
      for(int i = 0; i < coords.size() - 1; i++)
      {
        out << coords[i] << ", ";
      }
      out << coords[coords.size() - 1] << ")";
    }

    T & operator [] (int index)
    {
      if (in_bounds(index))
      {
        return coords[index];
      }
      else
      {
        cout << "Error: Index out of bounds" << endl;
        exit(-1);
      }
    }

    int dim()
    {
      return coords.size();
    }

  private:
    bool in_bounds(int i)
    {
      return i < coords.size() && i >= 0;
    }
    vector<T> coords;

};


template <typename T>
ostream & operator << (ostream & out, Point<T> pt)
{
  pt.print(out);
  return out;
}


#endif
