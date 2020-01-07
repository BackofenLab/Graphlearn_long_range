from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt



def genlin(x, a, b, c, d, g):
    return ( ( (a-d) / ( (1+( (x/c)** b )) **g) ) + d )

def logistic(x, p1,p2):
      return p1*np.log(x)+p2

def learncurve(xvals=[100,200,300],
        meanss=[(10,20,30),(5,15,25)],
        stdss=[ (20,20,20),(10,10,10)  ],
        labels = ['original + generated','generated','original'],colors='rgb'):

    func = logistic
    for means, stds,label,color in zip(meanss,stdss,labels,colors): 
        params_main,_ = curve_fit(func, xvals,means)
        params_upper,_ = curve_fit(func, xvals,[m+s for m,s in zip(means,stds)])
        params_lower,_ = curve_fit(func, xvals,[m-s for m,s in zip(means,stds)])

        x = range(50,max(xvals))
        print(params_main)
        plt.plot( x, [func(xx,*params_main) for xx in x ], label = label , color=color)
        plt.plot(xvals, means, 'o', color=color)
        plt.fill_between(x, [func(xx,*params_upper) for xx in x ],
                            [func(xx,*params_lower) for xx in x ], alpha=0.1, color=color)
    plt.legend()
    plt.show()
    
learncurve()
