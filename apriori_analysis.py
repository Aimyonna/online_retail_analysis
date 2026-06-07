import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

print("正在加载数据...")
df = pd.read_csv('/data/clean_retail.csv', encoding='utf-8')

# 数据转换：构建“购物篮”矩阵
print("正在构建法国区购物篮矩阵...")
basket = (df[df['Country'] == 'France']
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# 数值二值化
# 大于0的设为1，小于等于0的设为0
basket_sets = (basket > 0).astype(int)

# 剔除业务干扰项
if 'POSTAGE' in basket_sets.columns:
    basket_sets.drop('POSTAGE', inplace=True, axis=1)

# Apriori 算法：找出频繁项集
print("正在挖掘频繁项集...")
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)

# 生成关联规则：计算置信度与提升度
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# 按提升度(Lift)降序排列
rules = rules.sort_values('lift', ascending=False)

display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)
print("\n挖掘出的 Top 10 商品捆绑销售黄金规则：")
print(display_rules)