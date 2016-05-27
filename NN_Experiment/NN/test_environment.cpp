/*
 * Author: Aadil Ahamed
 * Created: 4/29/16
 * test_environment.h: Header file for test environment class
 */


#include "brute.h"
#include "brute_gpu.h"
#include "ann_imp.h"
#include "test_environment.h"
#include <assert.h>
#include <ctime>
#include <cmath>

void print_result(int N, int time1, int time2)
{
  cout << N << " " << time1 << " " << time2 << endl;
}

void print_result_v(NearestNeighbor<float> *impl1, NearestNeighbor<float> *impl2)
{
  cout << "Impl 1 - " << impl1->get_name() << endl;
  cout <<  "create time: " << (impl1->get_create_time()).count() << " ms " 
                        << " search time: " << (impl1->get_search_time()).count() << " ms " << endl;

  cout << "Impl 2 - " << impl2->get_name() << endl; 
  cout << "create time: " << (impl2->get_create_time()).count() << " ms " 
              << " search time: " << (impl2->get_search_time()).count() << " ms " << endl;
}

// Public Methods

 /*
  * compare: Compare running time of 2 implementations of Nearest Neighbor Search by performing N queries
  *          on N reference points where each point has dimension k
  */
void TestEnvironment::compare(NearestNeighbor<float> *impl1, NearestNeighbor<float> *impl2, int N, int k)
{
  //cout << "N = " << N << endl;
  vector<Point<float>> data (N);
  vector<Point<float>> queries (N);
  vector<int> res1 (N);
  vector<int> res2 (N);
  TestEnvironment::gen_data(data, k);
  TestEnvironment::gen_data(queries, k);
  impl1->nns(data, queries, res1);
  impl2->nns(data, queries, res2);
  if(!verify(res1, res2))
  {
    cout << "Results don't match !!!" << endl;
    exit(-1);
  }

  print_result(N, impl1->get_search_time().count(), impl2->get_search_time().count());

}

 /*
  * compare_growth: Compare running time of 2 implementations of Nearest Neighbor Search on input growing from 2^start till 2^end
  */
void TestEnvironment::compare_growth(NearestNeighbor<float> *impl1, NearestNeighbor<float> *impl2, int start, int end)
{
  int base = 2;
  int k = 2;
  cout << "#N impl1 impl2" << endl;
  for(int i = start; i < end; i++)
  {
    TestEnvironment::compare(impl1, impl2, pow(base, i), k);    
  }
}


// Private Methods

/*
 * verify: verifies whether 2 results are equal
 */
bool TestEnvironment::verify(const vector<int> &res1, const vector<int> &res2)
{
  if(res1.size() != res2.size())
  {
    return false;
  }
  for(int i = 0; i < res1.size(); i++)
  {
    if(res1[i] != res2[i])
    {
#ifdef DEBUG
      cout << "i = " << i << "res1 = " << res1[i] << "res2 = " << res2[i] << endl;
      cout << "i-1 = " << i-1 << "res1 = " << res1[i-1] << "res2 = " << res2[i-1] << endl;
#endif
      return false;
    }
  }
  return true;
}


/*
 * gen_data: generates N random points of dimension k in vector v where N is the size of v
 */
void TestEnvironment::gen_data(vector<Point<float>> &v, int k)
{
  assert(k > 0);
  srand(time(NULL));
  for(int i = 0; i < v.size(); i++)
  {
      vector<float> coords;
      for(int j = 0; j < k; j++)
      {
        coords.push_back(0 + 200.0*(rand() / (1.0 + RAND_MAX)));
      }
      v[i] = Point<float>(coords); 
  }
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

void test_verify_1()
{
  cout << "Verify Test 1" << endl;
  vector<int> res1 (100);
  vector<int> res2 (100);
  cout << "res1 == res2: " << TestEnvironment::verify(res1, res2) << endl;
  vector<int> res3 (50);
  vector<int> res4 (100);
  cout << "res3 == res4: " << TestEnvironment::verify(res3, res4) << endl;
  cout << endl;
}


void test_gen_data_1()
{
  int N = 10, k = 2;
  cout << "Generate Data Test 1" << endl;
  vector<Point<float>> v (N);
  TestEnvironment::gen_data(v, k);
  cout << "Gen Data: ";
  print_vector(v);
  cout << endl;
}

void test_compare_1()
{
  int N = 10000, k = 2;
  NearestNeighbor<float> *brute = new Brute<float>();
  NearestNeighbor<float> *ann = new AnnImp<float>();
  TestEnvironment::compare(brute, ann, N, k);
}

void test_compare_2()
{
  int N = 10000, k = 2;
  NearestNeighbor<float> *brute = new BruteGPU<float>();
  NearestNeighbor<float> *ann = new AnnImp<float>();
  TestEnvironment::compare(brute, ann, N, k);
}

void test_compare_growth_1()
{
  int k = 2, start = 1, end = 15;
  NearestNeighbor<float> *brute = new Brute<float>();
  NearestNeighbor<float> *ann = new AnnImp<float>();
  TestEnvironment::compare_growth(brute, ann, start, end);
}

void test_compare_growth_2()
{
  int k = 2, start = 1, end = 20;
  NearestNeighbor<float> *brute = new BruteGPU<float>();
  NearestNeighbor<float> *ann = new AnnImp<float>();
  TestEnvironment::compare_growth(brute, ann, start, end);
}


int main()
{
  /* initialize random seed: */
  srand (time(NULL));
  //test_gen_data_1();
  //test_verify_1();
  //test_compare_1();
  //test_compare_2();
  //test_compare_growth_1();
  test_compare_growth_2();
  return 0;
}

#endif
