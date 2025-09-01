import pandas as pd
df = pd.read_excel("Data.xlsx")

df_25 = df.sample(frac=0.25, random_state=42)
df_75 = df.drop(df_25.index)

df_25.to_csv("test.csv", index=False)
df_75.to_csv("training.csv", index=False)