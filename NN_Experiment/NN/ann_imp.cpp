/*
 * Author: Aadil Ahamed
 * Created: 4/28/16
 * ann_imp.cpp: Implementation of ann_imp class - Uses ANN library to implement nearest neighbor search
 */

#include "ann_imp.h"
#include <assert.h>


template <typename T>
void AnnImp<T>::ann_search( vector<Point<T>> &data, vector<Point<T>> &queries, vector<int> &result, ANNkd_tree *kd_tree, int k)
{
    assert (kd_tree);
    assert (k > 0);
    int dim = queries[0].dim();
    //dist_sq.resize(queries.size());
    ANNidxArray nnIdx = new ANNidx[k];
    ANNdistArray dists = new ANNdist[k];
    ANNpoint queryPt = annAllocPt(dim);
    chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();
    for(int i=0; i < queries.size(); i++) 
    {
        //Covert Point to ANNpoint
        for(int j=0; j < dim; j++) 
        {
            queryPt[j] = queries[i][j];
        }
        kd_tree->annkSearch(queryPt, k, nnIdx, dists);
        result[i] = nnIdx[0];
        //dist_sq[i] = dists[0];
    }
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
    NearestNeighbor<T>::search_time = chrono::duration_cast<chrono::milliseconds>( end - begin );
    annDeallocPt(queryPt);
    delete [] nnIdx;
    delete [] dists;
}

template <typename T>
void AnnImp<T>::nns( vector<Point<T>> &data, vector<Point<T>> &queries, vector<int> &result )
{
    assert (queries.size() == result.size());
    int dim = data[0].dim();
    ANNpointArray dataPts = annAllocPts( data.size(), dim );
    chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();
    // Covert input data vector to ANNpointArray 
    for(unsigned int i=0; i < data.size(); i++) 
    {
        for(int j=0; j < dim; j++ )
        {
            dataPts[i][j] = data[i][j];
        }
    }
    ANNkd_tree* kd_tree = new ANNkd_tree(dataPts, data.size(), dim);

    // Create time calculation
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
    NearestNeighbor<T>::create_time = chrono::duration_cast<chrono::milliseconds>( end - begin );

    //search
    ann_search(data, queries, result, kd_tree);
    delete kd_tree;
    annDeallocPts(dataPts);
    annClose();
}

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
  vector<int> result (queries.size());
  vector<Point<int>> final_res (queries.size());

  NearestNeighbor<int> *ann_imp = new AnnImp<int>();
  ann_imp->nns(data, queries, result);
  
  //cout << "result size = " << result.size() << endl;
  for(int i = 0; i < result.size(); i++)
  {
    //cout << "i = " << i << endl;
    final_res[i] = data[result[i]];
  }

  cout << "data: ";
  print_vector(data);
  cout << "queries: ";
  print_vector(queries);
  cout << "result: ";
  print_vector(final_res);
  cout << "search time = " << (ann_imp->get_search_time()).count() << " milliseconds" << endl;
  cout << "create time = " << (ann_imp->get_create_time()).count() << " milliseconds" << endl;
  cout << endl;
  
  delete ann_imp;
}

void test_nns_random()
{
  srand(time(NULL));
  int N = 4;
  float a = 0.0, b = 100.0;
  vector<Point<float>> data;
  vector<Point<float>> queries;

  for(unsigned int i=0; i < N; i++) {
      vector<float> coords;
      coords.push_back(a + b * (rand() / (1.0 + RAND_MAX)));
      coords.push_back(a + b * (rand() / (1.0 + RAND_MAX)));
      data.push_back(Point<float>(coords));
  }
  
  for(unsigned int i=0; i < N; i++) {
      vector<float> coords;
      coords.push_back(a + b * (rand() / (1.0 + RAND_MAX)));
      coords.push_back(a + b * (rand() / (1.0 + RAND_MAX)));
      queries.push_back(Point<float>(coords));
  }

  vector<int> result (queries.size());
  vector<Point<float>> final_res (queries.size());

  NearestNeighbor<float> *ann_imp = new AnnImp<float>();
  ann_imp->nns(data, queries, result);
  
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
  cout << "search time = " << (ann_imp->get_search_time()).count() << " milliseconds" << endl;
  cout << "create time = " << (ann_imp->get_create_time()).count() << " milliseconds" << endl;
  cout << endl;
  delete ann_imp;
}

int main()
{
  //test_nns_1();
  test_nns_random();
  return 0;
}

#endif


/*
template <typename T>
ANNkd_tree* AnnImp<T>::create_kdtree(ANNpointArray &dataPts, int dim)
{
    int dim = data[0].dim();
    ANNpointArray dataPts = annAllocPts( data.size(), dim );

    // Covert input data vector to ANNpointArray 
    for(unsigned int i=0; i < data.size(); i++) 
    {
        for(int j=0; j < dim; j++ )
        {
            dataPts[i][j] = data[i][j];
        }
    }
    ANNkd_tree* kd_tree = new ANNkd_tree(dataPts, data.size(), dim); 
    //annDeallocPts(dataPts);
    cout << "created kd_tree" << endl;
    return kd_tree;
}

template <typename T>
ANNpointArray & AnnImp<T>::to_annArray(vector<Point<T>> &data, int dim)
{
    ANNpointArray dataPts = annAllocPts( data.size(), dim );

    // Covert input data vector to ANNpointArray 
    for(unsigned int i=0; i < data.size(); i++) 
    {
        for(int j=0; j < dim; j++ )
        {
            dataPts[i][j] = data[i][j];
        }
    }
    return dataPts;
}
*/
