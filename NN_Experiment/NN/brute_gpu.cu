/*
 * Author: Aadil Ahamed
 * Created: 5/16/16
 * brute_gpu.cpp: Implementation of brute force nn on gpu
 */

#include "brute_gpu.h"

/*
template <typename T>
__device__ float serial_distance(Point<T> &p, Point<T> &q)
{
  float sdistance = 0;
  for(int i = 0; i < p.dim(); i++)
  {
    sdistance += square( p[i] - q[i] );
  }
  return sdistance;
}
*/

/**
* Kernel
* Executed on GPU
* Perform nearest neighbor search against data using queries
*/

template <typename T>
__global__ void nn(Point<T> *data, Point<T> *queries, int *result, int N)
{
  int global_id = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
  if (global_id < N)
  {
    Point<T> &qpoint = queries[global_id];
    int min_i = 0;
    float min_distance = Point<T>::distance(qpoint, data[0]);
    float t_distance = 0;
    for(int i = 1; i < N; i++)
    {
      t_distance = Point<T>::distance(qpoint, data[i]);
      if(t_distance < min_distance)
      {
        min_distance = t_distance;
        min_i = i;
      }
    }
    result[global_id] = min_i;
  }
}


template <typename T>
void BruteGPU<T>::nns( vector<Point<T>> &data, vector<Point<T>> &queries, vector<int> &result )
{
  int THREADS_PER_BLOCK = 512;
  int N = data.size();
  int size_pts = N * sizeof(Point<T>);
  int size_res = N * sizeof(int);
  int blocks = N/THREADS_PER_BLOCK + ((N % THREADS_PER_BLOCK) ? 1 : 0);
  Point<T> *d_data, *d_queries;
  int *d_results, *results;
 
  //Allocate space for result
  results = new int[N];

  // Allocate space for device copies of data and queries
  cudaMalloc((void **)&d_data, size_pts);
  cudaMalloc((void **)&d_queries, size_pts);
  cudaMalloc((void **)&d_results, size_res);

  cudaMemcpy(d_data, &data[0], size_pts, cudaMemcpyHostToDevice);
  cudaMemcpy(d_queries, &queries[0], size_pts, cudaMemcpyHostToDevice);
  
  chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();
  nn<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_queries, d_results, N);
  chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
  NearestNeighbor<T>::search_time = chrono::duration_cast<chrono::milliseconds>( end - begin );

  if(cudaGetLastError() != cudaSuccess)
  {
    printf("%s\n", cudaGetErrorString((cudaGetLastError())));
  }
  
  // Copy result back to host
  cudaMemcpy(results, d_results, size_res, cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_queries);
  cudaFree(d_results);
  //cout << "Results: " << endl;
  for(int i = 0; i < N; i++)
  {
    //cout << results[i] << " ";
    result[i] = results[i];
  }
  cout << endl;
  delete results;
}


template class BruteGPU<float>;
template class BruteGPU<int>;

//template __device__ float serial_distance(Point<float> &p, Point<float> &q);
//template __device__ float serial_distance(Point<int> &p, Point<int> &q);
template __global__ void nn(Point<float> *data, Point<float> *queries, int *result, int N);
template __global__ void nn(Point<int> *data, Point<int> *queries, int *result, int N);


#ifdef DEBUG

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
  //int N = 4;
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

  NearestNeighbor<int> *b = new BruteGPU<int>();
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
  NearestNeighbor<float> *b = new BruteGPU<float>();
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
  test_nns_1();
  //test_nns_random();
  return 0;
}

#endif
