/*
 * Author: Aadil Ahamed
 * Created: 4/29/16
 * test_environment.h: Header file for test environment class
 */


#include "test_environment.h"
#include <assert.h>
#include <ctime>

// Public Methods



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
        coords.push_back(0 + 100.0*(rand() / (1.0 + RAND_MAX)));
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


int main()
{
  test_gen_data_1();
  test_verify_1();
  return 0;
}

#endif
