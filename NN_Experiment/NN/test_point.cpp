/*
 * Author: Aadil Ahamed
 * Created: 4/25/16
 * test_point: Test Point Class
 */

#include <iostream>
#include "nearest_neighbor.h"


void test_print()
{
  cout << "Test Print" << endl;
  vector<float> v {0.0, 1.0};
  Point<float> p(v);
  cout << "P = " << p << endl;
  cout << endl;
}

void test_index()
{
  cout << "Test Print" << endl;
  vector<float> v {0.0, 1.0};
  Point<float> p(v);
  cout << "p[0] = " << p[0] << endl;
  p[0] = 0.5;
  cout << "p[0] = " << p[0] << endl;
  cout << endl;
}


int main()
{
  test_print();
  test_index();
  return 0;
}
