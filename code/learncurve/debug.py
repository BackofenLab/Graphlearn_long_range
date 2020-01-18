
import sys
import pickle 
func,iterab, loglevel = pickle.load( open( sys.argv[1], "rb" ) )  
import logging
logging.basicConfig(stream=sys.stdout, level=int(28)) 
logger = logging.getLogger(__name__)
print (func(iterab[2]))
from structout import gprint 
#for i in range(4): gprint(iterab[i], edgelabel=None)
