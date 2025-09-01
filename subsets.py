import pandas as pd
df = pd.read_excel("Data.xlsx")
# df has original data

df_25 = df.sample(frac=0.25, random_state=42)
df_75 = df.drop(df_25.index)
# split data into the respective subsets for training and testing

df_25.to_csv("test.csv", index=False)
df_75.to_csv("training.csv", index=False)
# save the data to a csv file for later use