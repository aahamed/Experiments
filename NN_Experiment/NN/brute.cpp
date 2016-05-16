/*
 * Author: Aadil Ahamed
 * Created: 4/25/16
 * brute.cpp: Implementation of Brute class
 */

#define square(x) ((x)*(x))

#include "brute.h"
#include <cstdlib>
#include <ctime>

// Distance Calculation (squared)

template <typename T>
float Brute<T>::serial_distance(Point<T> &p, Point<T> &q)
{
  float sdistance = 0;
  for(int i = 0; i < p.dim(); i++)
  {
    sdistance += square( p[i] - q[i] );
  }
  return sdistance;
}



template <typename T>
void Brute<T>::nns( vector<Point<T>> &data, vector<Point<T>> &queries, vector<int> &result )
{
  assert (queries.size() == result.size());
  NearestNeighbor<T>::create_time = chrono::milliseconds(0);
  chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();
  float min_distance, t_distance;
  int min_index = 0;
  for(int i = 0; i < queries.size(); i++)
  {
    min_index = 0;
    min_distance = serial_distance(queries[i], data[0]);
    for(int j = 0; j < data.size(); j++)
    {
      t_distance = serial_distance(queries[i], data[j]); 
      if(t_distance < min_distance)
      {
        min_index = j;
        min_distance = t_distance;
      }
    }
    result[i] = min_index;
  }
  chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
  NearestNeighbor<T>::search_time = chrono::duration_cast<chrono::milliseconds>( end - begin );
}

template class Brute<float>;




#ifdef DEBUG_BRUTE

template <typename T>
void print_vector(vector<T> &v)
{
  for(int i = 0; i < v.size(); i++)
  {
    cout << v[i] << " ";
  }
  cout << endl;
}


void test_nns_1()
{
  int N = 4;
  vector<int> v1 {0, 0};
  vector<int> v2 {0, 1};
  vector<int> v3 {0, 2};
  vector<int> v4 {0, 3};

  vector<int> w1 {0, -1};
  vector<int> w2 {1, 1};
  vector<int> w3 {1, 2};
  vector<int> w4 {1, 3};

  Point<int> p1(v1);
  Point<int> p2(v2);
  Point<int> p3(v3);
  Point<int> p4(v4);
  vector<Point<int>> data {p1, p2, p3, p4};

  Point<int> q1(w1);
  Point<int> q2(w2);
  Point<int> q3(w3);
  Point<int> q4(w4);
  vector<Point<int>> queries {q1, q2, q3, q4};
  vector<int> result(queries.size());
  vector<Point<int>> final_res(queries.size());

  NearestNeighbor<int> *b = new Brute<int>();
  b->nns(data, queries, result);

  for(int i = 0; i < result.size(); i++)
  {
    final_res[i] = data[result[i]];
  }

  cout << "data: ";
  print_vector(data);
  cout << "queries: ";
  print_vector(queries);
  cout << "result: ";
  print_vector(final_res);
  cout << "search time = " << (b->get_search_time()).count() << " milliseconds" << endl;
  cout << "create time = " << (b->get_create_time()).count() << " milliseconds" << endl;
  cout << endl;
}


void test_nns_random()
{
  srand(time(NULL));
  int N = 4;
  vector<Point<float>> data;
  vector<Point<float>> queries;

  for(unsigned int i=0; i < N; i++) {
      vector<float> coords;
      coords.push_back(0 + 100.0*(rand() / (1.0 + RAND_MAX)));
      coords.push_back(0 + 100.0*(rand() / (1.0 + RAND_MAX)));
      data.push_back(Point<float>(coords));
  }
  
  for(unsigned int i=0; i < N; i++) {
      vector<float> coords;
      coords.push_back(0 + 100.0*(rand() / (1.0 + RAND_MAX)));
      coords.push_back(0 + 100.0*(rand() / (1.0 + RAND_MAX)));
      queries.push_back(Point<float>(coords));
  }
  vector<int> result (queries.size());
  vector<Point<float>> final_res(queries.size());
  //cout << "starting nns" << endl;
  NearestNeighbor<float> *b = new Brute<float>();
  b->nns(data, queries, result);
  //cout << "ending nns" << endl;


  for(int i = 0; i < result.size(); i++)
  {
    final_res[i] = data[result[i]];
  }

  
  cout << "data: ";
  print_vector(data);
  cout << "queries: ";
  print_vector(queries);
  cout << "result: ";
  print_vector(final_res);
  
  cout << "search time = " << (b->get_search_time()).count() << " milliseconds" << endl;
  cout << "create time = " << (b->get_create_time()).count() << " milliseconds" << endl;
  cout << endl;
}

int main()
{
  //test_nns_1();
  test_nns_random();
  return 0;
}

#endif //DEBUG
