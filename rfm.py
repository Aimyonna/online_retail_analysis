import pandas as pd
import datetime as dt

df = pd.read_csv('/data/clean_retail.csv', encoding='utf-8')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 计算基准口
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# 计算 RFM
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Total Sales': 'sum'
}).reset_index()

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Total Sales': 'Monetary'
}, inplace=True)

print(rfm.head())
print(f"🎯 成功提取出的独立客户总数：{len(rfm)} 位")