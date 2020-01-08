from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt



def genlin(x, a, b, c, d, g):
    return ( ( (a-d) / ( (1+( (x/c)** b )) **g) ) + d )

def logistic(x, p1,p2):
      return p1*np.log(x)+p2

def logreversed(y,a,b):
    return np.exp((y-b)/a)

def learncurve(xvals=[100,200,300],
        meanss=[(10,20,30),(5,15,25)],
        stdss=[ (20,20,20),(10,10,10)  ],
        labels = ['original + generated','generated','original'],colors='rgb'):

    func = logistic
    funcrev = logreversed


    ###################
    # get params
    params = {}
    for means, stds,label in zip(meanss,stdss,labels): 
        main,_ = curve_fit(func, xvals,means)
        upper,_ = curve_fit(func, xvals,[m+s for m,s in zip(means,stds)])
        lower,_ = curve_fit(func, xvals,[m-s for m,s in zip(means,stds)])
        params[label] = [main,upper,lower]
    ##################3
    # make curve 
    for means, stds,label,color in zip(meanss,stdss,labels,colors): 
        x = range(50,max(xvals))
        params_main, params_upper, params_lower = params[label]
        plt.plot( x, [func(xx,*params_main) for xx in x ], label = label , color=color)
        plt.plot(xvals, means, 'o', color=color)
        plt.fill_between(x, [func(xx,*params_upper) for xx in x ],
                            [func(xx,*params_lower) for xx in x ], alpha=0.1, color=color)

    plt.legend()
    plt.show()
    

    ###########################
    # value increase: 
    print (params)
    po = params[labels[2]][0]  # orig
    print ( [funcrev(x ,*po) for x in meanss[0]] )


learncurve()
