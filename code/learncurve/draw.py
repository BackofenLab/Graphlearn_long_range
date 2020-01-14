from scipy.optimize import curve_fit
import basics as ba
import numpy as np
import matplotlib.pyplot as plt
import math 

def sigmoid(x, L ,x0, k,b):
    return  (L / (1 + np.exp(-k*(x-x0)))+b)

def fitsigmoid(_,x,y):
        p0 = [max(y), np.median(x),.5,min(y)] 
        return  curve_fit(sigmoid, x,y,p0)

def lin(x,a,b):
    return x*a+b


def learncurve(xvalss=[(100,200,300)],
        meanss=[(10,20,30),(5,15,25)],
        stdss=[ (20,20,20),(10,10,10)  ],
        rawdata=[(),(),()],
        labels = ['original + generated','original','generated'],
        colors='rgb'):

    func = sigmoid
    fit= fitsigmoid
    maxvals = max([x for xx in xvalss for x in xx])

    ###################
    # get params
    params = {}
    for means, stds,label,xvals,raw in zip(meanss,stdss,labels,xvalss,rawdata): 

        main,_ = fit(func, xvals,means)
        upper,_ = fit(func, xvals,[m+s for m,s in zip(means,stds)])
        lower,_ = fit(func, xvals,[m-s for m,s in zip(means,stds)])
        params[label] = [main,upper,lower]
    ##################3
    # make curve 
    for means, stds,label,color,raw,xvals in zip(meanss,stdss,labels,colors,rawdata,xvalss): 
        raw2 = list(zip(*raw))
        x = range(50,maxvals)

        params_main, params_upper, params_lower = params[label] ###  !!

        plt.plot( x, [func(xx,*params_main) for xx in x ], label = label , color=color)
        #plt.plot(xvals, means, 'o', color=color)
        props= {"color":color}
        plt.boxplot(raw2,positions=xvals,boxprops=props,
                widths=50,whiskerprops=props,capprops=props)
        plt.fill_between(x, [func(xx,*params_upper) for xx in x ],
                            [func(xx,*params_lower) for xx in x ], alpha=0.1, color=color)

    plt.ylabel("ROCAUC")
    plt.xlabel("Training instances")
    plt.legend()
    plt.show()
    

    ###########################
    # value increase: 
    print (params)
    po = params[labels[1]][0]  # orig
    #print ( [funcrev(x ,*po) for x in meanss[0]] )






a=[ [0.33433333333333337, 0.33066666666666666, 0.345, 0.369, 0.552,
0.6443333333333333], [0.6446666666666666, 0.6673333333333334, 0.67,
0.6696666666666667, 0.6686666666666667, 0.6666666666666666] ,
[0.33099999999999996, 0.3333333333333333, 0.33899999999999997,
0.3383333333333333, 0.3506666666666667, 0.3626666666666667]]

b=[ [0.014055445761538672, 0.010208928554075711, 0.01574801574802361,
0.018018509002319456, 0.03397057550292604, 0.01517307556898807]
,[0.015151090903151363, 0.017987650084309404, 0.020607442021431662,
0.011440668201153687, 0.008576453553512412, 0.01222929088522944] ,
[0.012247448713915879, 0.013021349989749726, 0.015895492023421807,
0.025315783394730028, 0.015584892970081268, 0.032714251057027466]]

x = [200,400,600,800,1000,1200]


#x,a,b,c = ba.loadfile("char_rnn.pickle")  
#x = [200,400,600,800,1200]
#print (x,a,b)
x,a,b,c = ba.loadfile("sav.sav")
learncurve(x,a,b,c )
