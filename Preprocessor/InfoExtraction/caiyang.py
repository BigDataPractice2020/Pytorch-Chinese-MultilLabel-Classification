import pandas as pd
def split(path):
    total = pd.read_excel(path, encoding='utf-8')
    set_1 = total.sample(n=1000, random_state=0, axis=0)
    set_2 =total.loc[~total.index.isin(set_1.index)]
    set_1.to_csv("m2_1000.csv")
    set_2.to_csv("m2_yu.csv")

split("input_m2_4-10/哈银M2.xlsx")