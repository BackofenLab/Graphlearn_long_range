
import sys
import pickle 
func,iterab, loglevel = pickle.load( open( sys.argv[1], "rb" ) )  
import logging
logging.basicConfig(stream=sys.stdout, level=int(28)) 
logger = logging.getLogger(__name__)
logger.log(29,"oi am here cloick me")
print (func(iterab[2]))
