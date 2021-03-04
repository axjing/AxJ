import baostock as bs
import pandas as pd
from plotly.offline import init_notebook_mode, iplot
# from IPython.display import HTML
import plotly.offline as py
import plotly.graph_objects as go
import cufflinks as cf
from fastai.tabular import *
import numpy as np
import re
import torch
from torch import nn
init_notebook_mode(connected=True)
cf.go_offline()
#### 登陆系统 ####
lg = bs.login()
def add_datepart(df, fldname, drop=True, time=False, errors="raise"):
    "Create many new columns based on datetime column."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld,
                      infer_datetime_format=True, errors=errors)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek','Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end',
            'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

#### 获取沪深A股历史K线数据 ####
# 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。
# 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
rs = bs.query_history_k_data_plus("sz.002648",
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
    start_date='2015-01-01', end_date='2020-4-14',
    frequency="d", adjustflag="3")

#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
df = pd.DataFrame(data_list, columns=rs.fields)

#### 结果集输出到csv文件 ####   
# result.to_csv("D:\\history_A_stock_k_data.csv", index=False)
print(df)

#### 登出系统 ####
bs.logout()



float_type = ['open','high','low','close','preclose','amount','pctChg']

for item in float_type:
    df[item] = df[item].astype('float')

df['amount'] = df['amount'].astype('int')
df['volume'] = df['volume'].astype('int')
df['turn'] = [0 if x == "" else float(x) for x in df["turn"]]
df['buy_flag'] = 10

def MA_next(df, date_idx, price_type, n): 
    return df[price_type][date_idx:date_idx+n].mean()

s_time = 2
m_time = 6
l_time = 15

for i in range(len(df)-l_time):
    if MA_next(df,i,'close',l_time)>MA_next(df,i,'close',m_time)*1.03>MA_next(df,i,'close',s_time)*1.03:
        df.loc[i, 'buy_flag'] = 2
    elif MA_next(df,i,'close',s_time)>MA_next(df,i,'close',m_time):
        df.loc[i, 'buy_flag'] = 0
    else:
        df.loc[i, 'buy_flag'] = 1
        df.loc[i, 'buy_flag'] = 1 + (MA_next(df,i,'close',m_time)-MA_next(df,i,'close',s_time))/MA_next(df,i,'close',s_time)
#     df.loc[i, 'buy_flag'] = 10*(MA_next(df,i,'close',m_time)+MA_next(df,i,'close',l_time)-2*MA_next(df,i,'close',s_time))/MA_next(df,i,'close',s_time)
        
df.tail()


fig = go.Figure(data=[go.Candlestick(x=df['date'],
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'],
                increasing_line_color= 'red', decreasing_line_color= 'green')
                     ])

fig.add_trace(go.Scatter(x=df['date'],y=df['buy_flag'], name='Flag'))

fig.update_layout(
    xaxis_range=['2017-01-01','2019-12-31'],
    yaxis_title='Price',
#     xaxis_rangeslider_visible=False,
)

py.iplot(fig, filename="stock-price")
py.plot(fig, filename="stock-price")


add_datepart(df, "date", drop=False)
seq_length = 90
train_df = df[seq_length:-seq_length]
# 丢掉不重要的特征
train_df = train_df.drop(['date','code','Is_month_end', 'Is_month_start', 'Is_quarter_end',
                          'Is_quarter_start', 'Is_year_end', 'Is_year_start','Dayofyear'],axis=1)
print(train_df)

def sliding_windows(data, label, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = label[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


from sklearn.preprocessing import MinMaxScaler
import numpy as np

y_scaler = MinMaxScaler()
x_scaler = MinMaxScaler()

#converting dataset into x_train and y_train
X = train_df.drop(['buy_flag'],axis=1).values
X = x_scaler.fit_transform(X)
Y = train_df['buy_flag']
Y = np.array(Y).reshape(-1,1)
Y = y_scaler.fit_transform(Y)

x, y = sliding_windows(X, Y, seq_length)

y_train,y_test = y[:int(y.shape[0]*0.8)],y[int(y.shape[0]*0.8):]
x_train,x_test = x[:int(x.shape[0]*0.8)],x[int(x.shape[0]*0.8):]

# lstm: seq, batch, feature
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataX = torch.Tensor(x.transpose(1,0,2))
dataY = torch.Tensor(y)
trainX = torch.Tensor(x_train.transpose(1,0,2))
trainY = torch.Tensor(y_train)
testX = torch.Tensor(x_test.transpose(1,0,2))
testY = torch.Tensor(y_test)
print(trainX.shape, trainY.shape)

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        # 不手动指定 h 和 c 的话，默认就是 0
#         h_0 = torch.zeros(
#             self.num_layers, x.size(0), self.hidden_size)
        
#         c_0 = torch.zeros(
#             self.num_layers, x.size(0), self.hidden_size)
        
        # Propagate input through LSTM
#         ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        ula, (h_out, _) = self.lstm(x)
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out


num_epochs = 15
learning_rate = 0.001

input_size = train_df.shape[1]-1 # The number of expected features in the input x
hidden_size = 300 # The number of features in the hidden state h
num_layers = 1 # Number of recurrent layers.

num_classes = 1 # output

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
lstm.train()
# lstm.to(device)
# trainX = trainX.to(device)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = lstm(trainX)
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    
    loss.backward()
    
    optimizer.step()
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

import plotly.graph_objects as go

lstm.eval()
lstm.to(torch.device('cpu'))
with torch.no_grad():
    dataY_pred = lstm(dataX)

dataY_pred = dataY_pred.data.numpy()
dataY_truth = dataY.data.numpy()

dataY_pred = y_scaler.inverse_transform(dataY_pred)
dataY_truth = y_scaler.inverse_transform(dataY_truth)


fig = go.Figure(go.Scatter(y=dataY_truth.flatten(),name='Ground Truth'))
fig.add_trace(go.Scatter(y=dataY_pred.flatten(),name='Predicted'))

fig.update_layout(
    shapes = [dict(
        x0=len(x_train), x1=len(x_train), y0=0, y1=1, xref='x', yref='paper',
        line_width=2)], #在图上划分训练集和测试集
    xaxis_rangeslider_visible=True,
)
py.iplot(fig, filename="stock-result")

import random
i = random.randint(0,testX.shape[1])
with torch.no_grad():
    y_pred = lstm(testX[:,i,::].reshape(testX.shape[0],1,testX.shape[2]))
print('预测值:{0}, 实际值:{1}'.format(y_pred.data.numpy(),testY[i].reshape(-1,1)))

