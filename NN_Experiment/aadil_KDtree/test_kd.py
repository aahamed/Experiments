"""
Author: Aadil Ahamed
test_kd.py: Class to test correctness of kd_tree
"""

from kd_tree import kd_tree
import nn_brute
import random
import time

DEBUG = False


def dump_input(array, filename):
    with open(filename, 'w') as tfile:
        for elem in array:
            tfile.write("{} {}".format(elem[0], elem[1]))
            tfile.write("\n")

def read_input(filename):
    array = []
    with open(filename, 'r') as tfile:
        for line in tfile:
            point = list(map(float, line.split(" ")))
            array.append(point)
    return array


class test_kd:
    
    def __init__(self):
        pass

    @staticmethod
    def get_randfloat(a=0, b=500):
        return random.uniform(a,b)

    @staticmethod
    def gen_data(n, a=0, b=500):
        """TODO"""
        data = [[test_kd.get_randfloat(a, b), test_kd.get_randfloat(a, b)] for i in range(n)]
        return data

    def test0(self):
        # data = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]]
        data = self.gen_data(10, 0, 15)
        queries = [[7,7]]
        tree = kd_tree(data)
        kd_res = tree.nns(queries[0])
        brute_res = nn_brute.nn(data, queries)
        if [kd_res] != brute_res:
            print("TEST 0: FAIL")
        else:
            if DEBUG:
                print("kd_res: ({0:.2f},{0:.2f})".format(*kd_res))
                print("brute_res: ({0:.2f}, {0:.2f})".format(*brute_res[0]))
            print("TEST 0: PASS")

    def test1(self):
        data = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]]
        queries = [[7,7], [4,5], [1,3], [2,9], [0, 40], [12, 3]]
        kd_res = kd_tree.nns_batch(data, queries)
        brute_res = nn_brute.nn(data, queries)
        if kd_res != brute_res:
            print("TEST 1: FAIL")
        else:
            print("TEST 1: PASS")
            if DEBUG:
                print("kd_res:", kd_res)
                print("brute_res:", brute_res)

    def test2(self, n=100):
        data = self.gen_data(n)
        queries = self.gen_data(n)
        kd_res = kd_tree.nns_batch(data, queries)
        brute_res = nn_brute.nn(data, queries)
        if kd_res != brute_res:
            print("Test 2: FAIL")
            print("ERROR: Mismatch in results")
            data_file = "data.txt"
            query_file = "query.txt"
            dump_input(data, data_file)
            dump_input(queries, query_file)
            test_kd.find_mismatch(brute_res, kd_res, queries)
        else:
            print("Test 2: PASS")
    
    def test3(self, n=300, max_depth=10):
        data = self.gen_data(n)
        queries = self.gen_data(n)
        kd_res = kd_tree.nns_batch(data, queries, max_depth)
        brute_res = nn_brute.nn(data, queries)
        if kd_res != brute_res:
            print("Test 3: FAIL")
        else:
            print("Test 3: PASS")

    def test4(self, query_file="query.txt", data_file="data.txt"):
        data = read_input(data_file)
        queries = read_input(query_file)
        kd_res = kd_tree.nns_batch(data, queries)
        brute_res = nn_brute.nn(data, queries)
        if kd_res != brute_res:
            print("Test 4: FAIL")
            print("ERROR: Mismatch in results")
            test_kd.find_mismatch(brute_res, kd_res, queries)
        else:
            print("Test 4: PASS")

    @staticmethod
    def find_mismatch(res1, res2, queries):
        if len(res1) != len(res2):
            print("Length mismatch")
        for i in range(len(res1)):
            if res1[i] != res2[i]:
                print("first mismatch: query = {3} res1[{0}] = {1}  res2[{0}] = {2} ".format(i, res1[i], res2[i], queries[i]))

    def time_test(self, n, max_depth=10):
        data = self.gen_data(n)
        queries = self.gen_data(n)
        # time brute force
        start = time.clock()
        brute_res = nn_brute.nn(data, queries)
        end = time.clock()
        brute_time = end - start

        # time kd_tree - no max depth
        tree = kd_tree(data)
        kd_res = []
        start = time.clock()
        for query in queries:
            kd_res.append(tree.nns(query))
        end = time.clock()
        kd_time = end - start

        # time kd_tree - with max depth
        tree = kd_tree(data, max_depth)
        kd_md_res = []
        start = time.clock()
        for query in queries:
            kd_md_res.append(tree.nns(query))
        end = time.clock()
        kd_md_time = end - start

        if brute_res == kd_res and kd_res == kd_md_res:
            print("brute time: {0:.2f}".format(brute_time))
            print("kd time: {0:.2f}".format(kd_time))
            print("kd with max_depth time: {0:.2f}".format(kd_md_time))
        else:
            flag1 = brute_res == kd_res
            flag2 = kd_res == kd_md_res
            print("ERROR: Mismatch in results")
            print("brute_res == kd_res: {}   kd_res == kd_md_res: {}".format(flag1, flag2))
            if not(flag1):
                test_kd.find_mismatch(brute_res, kd_res)
            else:
                test_kd.find_mismatch(kd_res, kd_md_res)
            
        
def main():
    t = test_kd()
    #t.test0()
    #t.test1()
    t.test2(1000)
    #t.test3()
    # t.time_test(5000, 10)
    #t.test4()


if __name__ == "__main__":
    main()
