/*
 * Author: Aadil Ahamed
 * Created: 4/29/16
 * test_environment.h: Header file for test environment class
 */

#ifndef TEST_ENVIRONMENT_H_
#define TEST_ENVIRONMENT_H

#include "nearest_neighbor.h"

//template <typename T>
class TestEnvironment
{
  public:
   /*
    * compare: Compare 2 implementations of Nearest Neighbor Search by performing N queries
    *          on N reference points where each point has dimension k
    */
   static void compare(NearestNeighbor<float> *impl1, NearestNeighbor<float> impl2, int N, int k);

   /*
    * compare_growth: Compare 2 implementations of Nearest Neighbor Search on input growing from 2^start till 2^end
    */
   static void compare_growth(NearestNeighbor<float> *impl1, NearestNeighbor<float> impl2, int start, int end);

    /*friends*/
    friend void test_verify_1();
    friend void test_gen_data_1();


  private:
    /*
     * verify: verifies whether 2 results are equal
     */
    static bool verify(const vector<int> &res1, const vector<int> &res2);

    /*
     * gen_data: generates N random points in vector v
     */
    static void gen_data(vector<Point<float>> &v, int k);
};


#endif // TEST_ENVIRONMENT 
