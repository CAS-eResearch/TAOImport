#!/usr/bin/env python
import sys
import pstats

def print_n(filename="taoconvert.prof"):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(20)

if __name__ == "__main__":
    args = sys.argv[1:]
    print_n(args[0])
    
    
