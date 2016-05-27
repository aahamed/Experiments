/*
 * Author: Aadil Ahamed
 * Created: 4/25/16
 * point.h: Header file for Point class
 */

#ifndef POINT_H_
#define POINT_H_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define square(x) ((x)*(x))
#define MAX_DIM 8

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda.h>

using namespace std;

template <typename T>
class Point
{
  public:
    Point()
    : dimension(0)
    {}

    Point(vector<T> in)
    : dimension(in.size())
    {
      if(in.size() > MAX_DIM)
      {
        exit(-1);
      }
      for(int i = 0; i < in.size(); i++)
      {
        coords[i] = in[i];
      }
      /*vector<T> *vp = new vector<T>(in);
      vector<T> &v = *vp;
      //cout << "v[0] = " << (v[0]) << endl;
      coords = &(v[0]);
      //cout << "coords[0] = " << coords[0] << endl;*/
    }
    Point(const Point<T> &p)
    : dimension(p.dim())
    {
      for(int i = 0; i < p.dim(); i++)
      {
        coords[i] = p.coords[i];
      }
      /*vector<T> *vp = new vector<T>();
      vector<T> &v = *vp;
      v.assign(p.coords, p.coords + p.dim());
      coords = &v[0];*/
    }
    ~Point(){}

    void print(ostream &out)
    {
      out << "(";
      for(int i = 0; i < dim() - 1; i++)
      {
        out << coords[i] << ", ";
      }
      out << coords[dim() - 1] << ")";
    }

    CUDA_CALLABLE_MEMBER T & operator [] (int index)
    {
      return coords[index];
    }

    CUDA_CALLABLE_MEMBER int dim() const
    {
      return dimension;
    }

    CUDA_CALLABLE_MEMBER static float distance(Point<T> &p, Point<T> &q)
    {
      float sdistance = 0;
      for(int i = 0; i < p.dim(); i++)
      {
       sdistance +=( p[i] - q[i] ) * ( p[i] - q[i] );
      }
      return sdistance;
    }

  private:
    CUDA_CALLABLE_MEMBER bool in_bounds(int i)
    {
      return i < dimension && i >= 0;
    }
    T coords[MAX_DIM];
    int dimension;

};


template <typename T>
ostream & operator << (ostream & out, Point<T> pt)
{
  pt.print(out);
  return out;
}


#endif
