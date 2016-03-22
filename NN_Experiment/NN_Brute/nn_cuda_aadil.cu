/*
 * nn_cuda_aadil.cu
 * Nearest Neighbor
 *
 */

#include <cstdio>
#include <stdio.h>
#include <cstdlib>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include <sys/time.h>
#include <string>
#include <fstream>
#include "cuda.h"

//#define DEBUG
#define DELTA 1E-6
#define THREADS_PER_BLOCK 512

using namespace std;

/*struct Point
{
    float coords[KDTREE_DIM];
};*/


typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

// Host Function Prototypes
float serial_distance(LatLong *p, LatLong *q);
int serial_min_index(float* distances, int N);
void serial_nn(LatLong *data, LatLong *queries, LatLong *result, int N);
bool match(LatLong *device, LatLong *host, int N);
double TimeDiff(timeval t1, timeval t2);
void loadVector(const char* filename, vector<LatLong> &v);
void write_result(const char* filename, int N, double time);

// Device Function Prototypes
__device__ float distance(LatLong *p, LatLong *q);
__device__ int min_index(float* distances, int N);
__global__ void nn(LatLong *data, LatLong *queries, LatLong *result, int N);


// Distance Calculation (squared)
float serial_distance(LatLong *p, LatLong *q)
{
  return (p->lat - q->lat)*(p->lat - q->lat) + (p->lng - q->lng)*(p->lng - q->lng);
}

// Find index with minimum distance
int serial_min_index(float *distances, int N)
{
  float min = distances[0];
  int min_i = 0;
  for(int i = 1; i < N; i++)
  {
    if(distances[i] < min)
    {
      min = distances[i];
      min_i = i;
    }
  }
  return min_i;
}

// Serial NN Search
void serial_nn(LatLong *data, LatLong *queries, LatLong *results, int N)
{
  float *distances = (float *)malloc(sizeof(float) * N);
  for(int i = 0; i < N; i++)
  {
    for(int j = 0; j < N; j++)
    {
      distances[j] = serial_distance(&queries[i], &data[j]); 
    }
    results[i] = data[serial_min_index(distances, N)];
  }
  free(distances);
}

// Checks if serial and parallel results match
bool match(LatLong *device, LatLong *host, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(serial_distance(&device[i], &host[i]) > DELTA)
    {
      printf("first mismatch at index %d\n", i);
      return false;
    }
  }
  return true;
}

//Calculate the time difference between t1 and t2 -> outputs in ms
double TimeDiff(timeval t1, timeval t2)
{
    double t;
    t = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    t += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    return t;
}

// Load data from filename into vector
void loadVector(const char* filename, vector<LatLong> &v)
{
  ifstream file(filename);
  char* pEnd;
  string value;
  LatLong l;
  if(file.is_open())
  {
    while( getline(file, value) )
    {
      l.lat = strtof(value.c_str(), &pEnd);
      l.lng = strtof(pEnd, NULL);
      v.push_back(l);
    }
  }
  file.close();
#ifdef DEBUG2
  for(int i = 0; i < v.size(); i++)
  {
    printf("%.2f %.2f\n", v[i].lat, v[i].lng);
  }
#endif
}

void write_result(const char* filename, int N, double time)
{
  ofstream file;
  file.open(filename, ios::out | ios::app);
  if(file.is_open())
  {
    file << N << "\t" << time << "\n";
  }
  file.close();
}


// Distance calculation (squared)
__device__ float distance(LatLong *p, LatLong *q)
{
  return (p->lat - q->lat)*(p->lat - q->lat) + (p->lng - q->lng)*(p->lng - q->lng);
}

// Find index with minimum distance
__device__ int min_index(float* distances, int N)
{
  float min = distances[0]; //inefficient ?
  int min_i = 0;
  for(int i = 1; i < N; i++)
  {
    if(distances[i] < min)
    {
      min = distances[i];
      min_i = i;
    }
  }
#ifdef DEBUG
  if(threadIdx.x == 0)
  {
    printf("Min index for thread 0 = %d  value = %.2f\n", min_i, min);
  }
#endif
  return min_i;
}


/**
* Kernel - deprecated
* Executed on GPU
* Perform nearest neighbor search against data using queries
*/
__global__ void nn_dep(LatLong *data, LatLong *queries, LatLong *result, int N)
{
  int global_id = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
  if (global_id < N)
  {
    LatLong* qpoint = &queries[global_id];
    float* distances = (float*)malloc(sizeof(float) * N);
    for(int i = 0; i < N; i++)
    {
      distances[i] = distance(qpoint, &data[i]);
    }
    result[global_id] = data[min_index(distances, N)];
    free(distances);
  }

}


/**
* Kernel
* Executed on GPU
* Perform nearest neighbor search against data using queries
*/
__global__ void nn(LatLong *data, LatLong *queries, LatLong *result, int N)
{
  int global_id = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
  if (global_id < N)
  {
    LatLong* qpoint = &queries[global_id];
    int min_i = 0;
    float min_distance = distance(qpoint, &data[0]);
    float t_distance = 0;
    for(int i = 1; i < N; i++)
    {
      t_distance = distance(qpoint, &data[i]);
      if(t_distance < min_distance)
      {
        min_distance = t_distance;
        min_i = i;
      }
    }
    result[global_id] = data[min_i];
  }

}

// Main body
int main()
{
  vector <LatLong> data;
  vector <LatLong> queries;
  LatLong *d_data, *d_queries, *d_results, *results, *serial_results;
  timeval t1, t2;
  double elapsed_gpu, elapsed_serial;
 
  const char *data_file = "data/data.txt";
  const char *query_file = "data/query.txt";
  const char *out_file = "output/brute_out.txt";
  loadVector(data_file, data);
  loadVector(query_file, queries);
  if(queries.size() != data.size())
  {
    printf("size of query set does not match size of data set\n");
    return 1;
  }

  int N = queries.size();
  int blocks = N/THREADS_PER_BLOCK + ((N % THREADS_PER_BLOCK) ? 1 : 0);
  int size = N * sizeof(LatLong);

  /* initialize random seed: */
  srand (time(NULL));

  /*
  for(int i = 0; i < data.size(); i++)
  {
    data[i].lat = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
    data[i].lng = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
  }
  for(int i = 0; i < queries.size(); i++)
  {
    queries[i].lat = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
    queries[i].lng = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
  }*/
  
  //Allocate space for result
  results = (LatLong *)malloc(size);
  serial_results = (LatLong *)malloc(size);

  // Allocate space for device copies of data and queries
  cudaMalloc((void **)&d_data, size);
  cudaMalloc((void **)&d_queries, size);
  cudaMalloc((void **)&d_results, size);

  // Copy inputs to device
  cudaMemcpy(d_data, &data[0], size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_queries, &queries[0], size, cudaMemcpyHostToDevice);

  gettimeofday(&t1, NULL);
  //Launch kernel for GPU NN Search
  nn<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_queries, d_results, N);
  if(cudaGetLastError() != cudaSuccess)
  {
    printf("%s\n", cudaGetErrorString((cudaGetLastError())));
  }
  gettimeofday(&t2, NULL);
  elapsed_gpu = TimeDiff(t1, t2); 
  // Copy result back to host
  cudaMemcpy(results, d_results, size, cudaMemcpyDeviceToHost);

  gettimeofday(&t1, NULL);

#ifdef DEBUG
  //Do Serial NN Search
  serial_nn(&data[0], &queries[0], serial_results, N);
  gettimeofday(&t2, NULL);
  elapsed_serial = TimeDiff(t1, t2);

  for(int i = 0; i < 5; i++)
  {
    printf("query: (%.2f, %.2f) gpu_nn: (%.2f, %.2f) ser_nn(%.2f, %.2f)\n", 
        queries[i].lat, queries[i].lng, results[i].lat, results[i].lng, 
        serial_results[i].lat, serial_results[i].lng);
  }
  
  if(match(results, serial_results, N))
  {
    printf("Host and Device Results match !\n");
    printf("GPU Running Time: %.3f\n", elapsed_gpu);
    printf("Serial Running Time: %.3f\n", elapsed_serial);
    printf("Speedup on GPU: %.2f\n", elapsed_serial/elapsed_gpu);
    write_result(out_file, N, elapsed_gpu);
  }
  else
  {
    printf("Host and Device Results DO NOT match !\n");
  }

#endif
  write_result(out_file, N, elapsed_gpu);
  //Cleanup
  free(results);
  cudaFree(d_data);
  cudaFree(d_queries);
  cudaFree(d_results);
  return 0;
}

