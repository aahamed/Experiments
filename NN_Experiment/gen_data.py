"""
Author: Aadil Ahamed
Date: 2/23/16
gen_data.py: TODO
"""
import random
import argparse

def gen_data(filename, N):
  data = [get_randcoord() for i in range(N)]
  with open(filename, 'w') as data_file:
    for i in range(len(data)):
      coord = data[i]
      data_file.write(coord[0] + " " + coord[1]  + "\n")

def get_randint(a=0, b=1000000000):
  return random.randint(a, b)

def get_randfloat(a=0, b=500):
  return random.uniform(a, b)

def get_randcoord():
  return (str(get_randfloat()), str(get_randfloat()))


def main():
    data_file = 'data/data.txt'
    query_file = 'data/query.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="Number of records to generate", type=int)
    args = parser.parse_args()
    N = args.N
    gen_data(data_file, N)
    gen_data(query_file, N)

if __name__ == "__main__":
    main()
