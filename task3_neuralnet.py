from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import LinearRegression

# df = pd.read_csv('data_challenge_index_prices.csv')
# # print(df)
# df.iloc[: , 1]-=df.iloc[:, 2]
# #df.loc[0]
# for i in range(199999):
#     prev = df.loc[i+1]
#     (df.loc[i])= -(df.loc[i]) + prev
# df = df.drop(df.index[-1])
# #print(df)
# df.to_csv('index_returns.csv', index=False)



df_input = pd.read_csv("returns.csv")
df_output = pd.read_csv("index_returns.csv")

# print(df_input)
# print(df_output)

sec1 = ['8', '12', '27', '29', '30', '36', '40', '41', '44', '46', '50', '52', '55', '57', '59', '69', '71', '73', '74', '77', '80', '84', '92', '96', '99']
sec2 = ['2', '3', '6', '7', '9', '16', '17', '19', '23', '28', '32', '33', '42', '47', '51', '53', '58', '60', '63', '66', '72', '81', '87', '91', '98']
sec3 = ['0', '1', '4', '10', '11', '14', '18', '20', '21', '22', '25', '26', '38', '39', '43', '45', '49', '54', '67', '75', '78', '79', '82', '88', '90']
sec4 = ['5', '13', '15', '24', '31', '34', '35', '37', '48', '56', '61', '62', '64', '65', '68', '70', '76', '83', '85', '86', '89', '93', '94', '95', '97']
sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0


X = df_input.to_numpy()
# print(X)
# print(df_output.loc[: ,"0"])
for ind in range(15):
    y = df_output.loc[:, str(ind)].to_numpy()
    reg = LinearRegression()
    reg.fit(X, y)
    # print(reg.coef_)
    arr = [i for i in range(1,101)]
    # plt.scatter(reg.coef_ , arr )
    # plt.plot()
    for i in range(100):
        if str(i) in sec1:
            sum1 += (reg.coef_)[i]
        if str(i) in sec2:
            sum2 += (reg.coef_)[i]
        if str(i) in sec3:
            sum3 += (reg.coef_)[i]
        if str(i) in sec4:
            sum4 += (reg.coef_)[i]
    print(sum1,sum2,sum3,sum4)
    sum1 = 0
    sum2=0
    sum3=0
    sum4=0



