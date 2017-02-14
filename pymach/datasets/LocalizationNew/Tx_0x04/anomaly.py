
# coding: utf-8

# ## RAW data Beacons

# In[145]:

import csv
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


#NUMBER = 15

for i in range(1,16):
    NUMBER = i
    name = 'p'+str(NUMBER)
    with open(name, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line for line in stripped if line)
        grouped = itertools.izip(*[lines] * 2)
        with open(name+'.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('Beacon', 'RSSI'))
            writer.writerows(grouped)


    df = pd.read_csv(name+'.csv')
    #print df.shape
    #df.head(10)


    condition = ''
    with open('devices', 'r') as inFile:
        stripped = (line.strip() for line in inFile)
        
        for x in stripped:
            condition += '(Beacon == "' + str(x) + '")' + ' | '
        condition = condition[:-3]
        
    #print condition


    # In[148]:

    df = df.query(condition)

    #df.shape


    ## In[149]:

    #df.Beacon.unique()


    # In[150]:

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df['Beacon'] = encoder.fit_transform(df['Beacon'])

    #df.head(10)


    ## In[151]:

    #df.Beacon.unique()


    ## In[152]:

    #df.groupby('Beacon').size()


    # In[153]:

    # for (a,b) in df.groupby('Beacon'):
    #     print (a,b)
    import pandas as pd

    l = {}
    d = {}
    for b, rssi in df.groupby('Beacon'):
    #     print b, type(rssi)
        l[b] = rssi
        #d[b] = {b:l[b]['RSSI']}


    # df2 = pd.DataFrame(list(l), columns=range(5) )
    df2 = pd.DataFrame({0:l[0]['RSSI'], 1:l[1]['RSSI'], 2:l[2]['RSSI'], 3:l[3]['RSSI'],
                       4:l[4]['RSSI']})
    # df2 = pd.DataFrame(d.items(), columns=list('01234'))
    #df2.head(10)


    # In[154]:

    #get_ipython().magic(u'matplotlib inline')
    #import matplotlib.pyplot as plt
    #plt.figure(figsize=(10.8, 3.6))
    #df2.hist(color=[(0.196, 0.694, 0.823)])
    ## df2.plot(kind="box" , subplots=True, layout=(2,3), sharex=False, sharey=False)
    #plt.show()


    # In[155]:

    #df2.describe()


    ## In[156]:

    #df2.head(15)


    # In[219]:

    dfx = df2.copy()
    dfx.head()


    # In[220]:

    dfx = dfx.fillna(method='ffill')
    dfx = dfx.dropna()
    print(len(dfx))
    #dfx.head(20)


    # In[221]:

    # dfx2 = dfx.drop_duplicates([0,1,2,3,4])
    dfx = dfx.drop_duplicates()
    print len(dfx)
    #dfx.head()

    #dfx = dfx.drop_duplicates([1,2,3,4])
    #dfx = dfx.drop_duplicates([0,1,2,3])
    #dfx = dfx.drop_duplicates([1,2,3])

    # In[222]:

    scaler = StandardScaler()
    dfx = scaler.fit_transform(dfx.ix[:, 0:4])
    dfx = pd.DataFrame(dfx)
    dfx.head()


    # In[223]:

    print(len(dfx))


    # In[224]:

    #from sklearn.covariance import EllipticEnvelope



    # In[225]:

    #len(outliers[0])

    dfx = dfx[dfx.apply(lambda x: np.abs(x - x.median()) / x.std() < 3).all(axis=1)]


    # In[227]:

    print(len(dfx))
    dfx.head()


    # In[165]:

    dfx = scaler.inverse_transform(dfx)
    dfx = pd.DataFrame(dfx)


    # In[166]:

    vector = NUMBER*np.ones((len(dfx),), dtype=np.int)
    dfx = dfx.assign(position=pd.Series(vector).values)
    #dfx.columns = ['b1','b2','b3','b4','b5','position']


    # In[167]:

    dfx.head()


    # ## Write csv

    # In[168]:

    dfx.to_csv(name+'_new'+'.csv', index=False)


# In[ ]:




# In[ ]:




# In[ ]:



