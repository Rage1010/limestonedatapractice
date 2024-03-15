import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# df = pd.read_csv('data_challenge_stock_prices.csv')
# print(df)
# #df.iloc[: , 1]-=df.iloc[:, 2]
# #df.loc[0]
# for i in range(199999):
#     prev = df.loc[i+1]
#     (df.loc[i])= -(df.loc[i]) + prev
# df = df.drop(df.index[-1])
# #print(df)
# df.to_csv('returns.csv', index=False)
df = pd.read_csv('returns.csv')
#print (df)
correlation_threshold = 0.19
dff = df.corr()
heat_map = sns.heatmap(dff.corr())
plt.show()
l=[{1}]
for i in range(100):
    obj = dff.loc[str(i)]
    for j in range(100):
        if j==i:
            continue
        if (obj[j]>correlation_threshold):
            a = False
            for n in l:
                if i in n and j in n:
                    a=True
                if i in n and j not in n:
                    n.add(j)
                    a=True
                if i not in n and j in n:
                    n.add(i)
                    a = True
            if not a:
                l.append({i,j})
for i in range(100):
    li=[]
    b=0
    for k in l:
        if i in k:
            li.append(b)
        b+=1
    if len(li)>0:
        sd=li[0]
        for t in li:
            l[sd].update(l[t])
        for t in range(len(li)):
            ind = len(li)-1-t
            rg = li[ind]
            if sd!=rg:
                l.pop(rg)

for i in l:
    print(len(i))
l[0].add(4)
l[0].add(10)
print(l) 
                
# for y in range(100):
#     a = False
#     for h in l:
#         a = a or y in h
#     if not a:
#         print(y,':')
#         for h in l:
#             sum =0
#             for x in h:
#                 sum += (dff.loc[str(y)])[x]
#             print(sum)
# print(dff)


