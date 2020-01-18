
import pandas as pd 


stuff = pd.read_csv("tmp", index_col=0, header = None)


print(stuff)
print(stuff.index)
snnid=["FCD" in x for x in stuff.index]
stuff[snnid]*=-1
stuff.loc['weight']-=300
stuff.loc['weight']= stuff.loc['weight'].abs()
stuff.loc['weight']*=-1
print(stuff.to_latex())
mo = stuff.rank(axis=1 ,ascending=False) 



asd= pd.DataFrame(mo)
mo.loc['average'] = asd.mean(numeric_only=True, axis=0)
mo.loc['median'] = asd.median(numeric_only=True, axis=0)

mo.loc[:,'rank'] =  mo.loc[:,4:].min(axis=1)

#mo.rename('median','loo')
z=mo.index
z=[zz+"uparrow" if "FCD" not in zz else zz+"downarrow" for zz in z ]
mo.index=z
print(mo.to_latex())


#python3 ranker.py  | sed s/uparrow/\\\\uparrow/ | sed s/downarrow/\\\\downarrow/


