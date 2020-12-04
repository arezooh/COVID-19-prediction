from sys import argv
import sys
import os
import subprocess
from pexecute.process import ProcessLoom

base_r = 6
r = 10

def Run(i):
    subprocess.call("python ./"+ str(r) +"/prediction.py "+str(i), shell=True)

def main():
    for i in range(19 - base_r - r + 7):
        print(200 * '*')
        print(i)
        Run(i)



if __name__ == "__main__":

    main()
