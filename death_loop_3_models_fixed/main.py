from sys import argv
import sys
import os
import subprocess

def Run(r, test_point):
    subprocess.call("python ./" + str(r) + "/prediction.py " + str(test_point), shell=True)

def main():
    if int(argv[1]) >= 0:
        for r in range(1, 11):
            test_point = 10 - r + int(argv[1])
            print(100*'*')
            print('r =', r, ', test_point =', test_point)
            Run(r, test_point)
    else:
        for r in range(1, 11 + int(argv[1])):
            test_point = 10 - r + int(argv[1])
            print(100*'*')
            print('r =', r, ', test_point =', test_point)
            Run(r, test_point)


if __name__ == "__main__":
    main()
