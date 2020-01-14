
import pandas as pd 


stuff = pd.read_csv("tmp", index_col=0, header = None)


print(stuff)
print(stuff.index)
snnid=["SNN" in x for x in stuff.index]
stuff[snnid]*=-1
stuff.loc['weight']*=-1
stuff.loc['weight']+=300
print(stuff)
mo = stuff.rank(axis=1 ,ascending=False) 
mo.loc['average'] = mo.mean(numeric_only=True, axis=0)
mo.loc['median'] = mo.median(numeric_only=True, axis=0)

print(mo)





