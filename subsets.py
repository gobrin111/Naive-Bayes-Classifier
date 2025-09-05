import pandas as pd
import math


class Classifier:
    def __init__(self):
        self.benign_variance = None
        self.malignant_variance = None
        self.benign_mean = None
        self.malignant_mean = None
        self.df_75_zeros = None
        self.df_75_ones = None
        self.df_75 = None
        self.df_25 = None
        self.df = None
        self.accuracy = None

    def train(self, file):
        self.df = pd.read_excel(file)
        # df has original data

        # split data into the respective subsets for training and testing
        self.df_25 = self.df.sample(frac=0.25, random_state=42) # random state set to 42 so the 75 25 split is the same every time program runs
        self.df_75 = self.df.drop(self.df_25.index)

        # sort training data into two with the malignant and the benign
        self.df_75_ones = self.df_75[self.df_75.iloc[:, 30] == 1]
        self.df_75_zeros = self.df_75[self.df_75.iloc[:, 30] == 0]

        # uses panda function to get the mean of each column except for the last one
        self.malignant_mean = self.df_75_zeros.iloc[:, :30].mean().values
        self.benign_mean = self.df_75_ones.iloc[:, :30].mean().values

        # uses panda function to get the variance of each column except for the last one
        self.malignant_variance = self.df_75_zeros.iloc[:, :30].var().values
        self.benign_variance = self.df_75_ones.iloc[:, :30].var().values

    def testRow(self, currentRow):
        # holders for the value of each class
        malignantVal = 1
        benignVal = 1
        for i in range(30):
            # calculate using probability distribution
            malignantVal *= (1 / ((math.sqrt(2 * math.pi)) * math.sqrt(self.malignant_variance[i]))) * (math.e ** (-(((currentRow[i] - self.malignant_mean[i]) ** 2) / (2 * self.malignant_variance[i]))))
            benignVal *= (1 / ((math.sqrt(2 * math.pi)) * math.sqrt(self.benign_variance[i]))) * (math.e ** (-(((currentRow[i] - self.benign_mean[i]) ** 2) / (2 * self.benign_variance[i]))))

        # then compare the two values and select the result
        if malignantVal > benignVal:
            return 0
        else:
            return 1

    def runTest(self):
        # measure accuracy with the actual answer
        self.accuracy = 0
        for i, row in self.df_25.iterrows():
            result = self.testRow(row)
            if result == row[30]:
                self.accuracy += 1
        self.accuracy /= len(self.df_25)
        print("Accuracy: ", self.accuracy)



# this is here to run the functions and input the file
gnbc = Classifier()
gnbc.train("Data.xlsx") # just change the file name here to use other files for testing
gnbc.runTest()

# run X example from project sheet
# change X variable if needed for testing
X = [13.0, 15.0, 85.0, 500.0, 0.1, 0.15, 0.1, 0.05, 0.2, 0.08, 0.5, 1.5, 4.0, 70.0, 0.01, 0.02,
     0.02, 0.01, 0.015, 0.002, 14.0, 20.0, 90.0, 600.0, 0.2, 0.25, 0.2, 0.1, 0.3, 0.1]

if gnbc.testRow(X):
    print("Single test is benign")
else:
    print("Single test is malignant")
