#!/usr/bin/env python
import pickle 
from structout import gprint 
import sys

g = pickle.load( open(sys.argv[1], "rb" ) )
gprint(g, edgelabel=None)
