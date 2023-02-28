import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from statsmodels.graphics.mosaicplot import mosaic
from itertools import product
import plotly
import plotly.express as px


###################################
# Importing data into a DataFrame #
#         and cleaning it         #
###################################
dataList = []

# with open("adult.data", newline="") as csvfile:
#    csvreader = csv.reader(csvfile, delimiter=",", quotechar="|")
#    for row in csvreader:
#        dataList.append(row)


df = pd.read_csv("adult.data", sep=",", header=None, skipinitialspace=True)
df.columns = [
    "Age",
    "Workclass",
    "Fnlwgt",
    "Education",
    "Education Num",
    "Marital Status",
    "Occupation",
    "Relationship",
    "Race",
    "Sex",
    "Capital Gain",
    "Capital Loss",
    "Hours Per Week",
    "Native Country",
    "Pay",
]
# result = df.head(5)
# print(result)

indicesToDelete = df[
    (df["Age"] == "?")
    | (df["Workclass"] == "?")
    | (df["Fnlwgt"] == "?")
    | (df["Education Num"] == "?")
    | (df["Marital Status"] == "?")
    | (df["Occupation"] == "?")
    | (df["Relationship"] == "?")
    | (df["Race"] == "?")
    | (df["Sex"] == "?")
    | (df["Capital Gain"] == "?")
    | (df["Capital Loss"] == "?")
    | (df["Hours Per Week"] == "?")
    | (df["Native Country"] == "?")
    | (df["Pay"] == "?")
].index


df.drop(indicesToDelete, inplace=True)

####################
#   User story 1   #
# Histogram of Age #
####################
# print("min age: ", min(df["Age"]))
# print("max age: ", max(df["Age"]))
fig, axs = plt.subplots(1, 1, figsize=(5, 4))
axs.hist(df["Age"], bins=20, rwidth=0.9, color="#607c8e")
plt.xlabel("Age")
plt.ylabel("Counts")
plt.title("Histogram of Ages for Census Data")
plt.show()

########################
#     User Story 2     #
# Education vs. Salary #
########################
educationPayDict = {}
educationCountDict = {}
for idx, row in df.iterrows():
    # print(row)
    key = row["Education Num"]
    if row["Education Num"] not in educationCountDict:
        educationCountDict[row["Education Num"]] = 0
        educationPayDict[row["Education Num"]] = 0
    if row["Pay"] == ">50K":
        educationPayDict[row["Education Num"]] += 1

    educationCountDict[row["Education Num"]] += 1

sortedKeys = list(educationPayDict.keys())
sortedKeys.sort()
# print(sortedKeys)
educationPayRatio = {
    i: educationPayDict[i] * 100 / educationCountDict[i] for i in sortedKeys
}
# print(educationPayRatio)
plt.plot(sortedKeys, educationPayRatio.values())
plt.xlabel("Grade Level")
plt.ylabel("Percentage Making > 50K")
plt.title("Percentage Making > 50K vs Grade Level")
plt.show()

##################
#  User Story 3  #
# Age vs. Salary #
##################
agePayDict = {}
ageCountDict = {}
for idx, row in df.iterrows():
    # print(row)
    key = row["Age"]
    if key not in ageCountDict:
        ageCountDict[key] = 0
        agePayDict[key] = 0
    if row["Pay"] == ">50K":
        agePayDict[key] += 1

    ageCountDict[key] += 1

# print(agePayDict)
# print(ageCountDict)

sortedKeys = list(agePayDict.keys())
sortedKeys.sort()
# print(sortedKeys)
agePayRatio = {i: agePayDict[i] * 100 / ageCountDict[i] for i in sortedKeys}
# print(agePayRatio)
plt.plot(sortedKeys, agePayRatio.values())
plt.xlabel("Age")
plt.ylabel("Percentage Making > 50K")
plt.title("Percentage Making > 50K vs Age")
plt.show()

###################################
#          User Story #4          #
# Mosaic plot of gender vs salary #
###################################

crosstable = pd.crosstab(df["Sex"], df["Pay"])
# print(crosstable)
props = {}
props[("Male", ">50K")] = {"facecolor": "lightblue", "edgecolor": "white"}
props[("Male", "<=50K")] = {"facecolor": "lightblue", "edgecolor": "white"}
props[("Female", ">50K")] = {"facecolor": "lightpink", "edgecolor": "white"}
props[("Female", "<=50K")] = {"facecolor": "lightpink", "edgecolor": "white"}
labelizer = lambda k: {
    ("Male", ">50K"): 6662,
    ("Female", ">50K"): 1179,
    ("Male", "<=50K"): 15128,
    ("Female", "<=50K"): 9592,
}[k]
mosaic(
    df,
    ["Sex", "Pay"],
    labelizer=labelizer,
    properties=props,
    title="Mosaic Plot of Pay for Males vs Females",
)
plt.show()

####################################
#          User Story #5           #
# Choropleth Map of Pay by Country #
####################################

countryPayDict = {}
countryCountDict = {}
for idx, row in df.iterrows():
    # print(row)
    key = row["Native Country"]
    if key not in countryCountDict:
        countryCountDict[key] = 0
        countryPayDict[key] = 0
    if row["Pay"] == ">50K":
        countryPayDict[key] += 1

    countryCountDict[key] += 1

# print(countryPayDict)
# print(countryCountDict)

sortedKeys = list(countryPayDict.keys())
sortedKeys.sort()
# print(sortedKeys)
countryPayRatio = {i: countryPayDict[i] * 100 / countryCountDict[i] for i in sortedKeys}
del countryPayRatio["Columbia"]
del countryPayRatio["Scotland"]

# print(countryPayRatio)

choroplethDf = pd.DataFrame(
    {
        "Country": [
            "KHM",
            "CAN",
            "CHN",
            "CUB",
            "DOM",
            "ECU",
            "SLV",
            "GBR",
            "FRA",
            "DEU",
            "GRC",
            "GTM",
            "HTI",
            "NLD",
            "HND",
            "HKG",
            "HUN",
            "IND",
            "IRN",
            "IRL",
            "ITA",
            "JAM",
            "JPN",
            "LAO",
            "MEX",
            "NIC",
            "GUM",
            "PER",
            "PHL",
            "POL",
            "PRT",
            "PRI",
            "ZAF",
            "TWN",
            "THA",
            "TTO",
            "USA",
            "VNM",
            "YUG",
        ],
        "Pay Percentage": list(countryPayRatio.values()),
    }
)
# print(choroplethDf)

fig = px.choropleth(
    choroplethDf,
    locations="Country",
    color="Pay Percentage",
    color_continuous_scale=px.colors.sequential.Plasma,
)
fig.show()

#########################
# Binary Classification #
#########################


df.columns = [c.replace(" ", "_") for c in df.columns]
df["Pay"].replace(["<=50K", ">50K"], [0, 1], inplace=True)
df["Sex"].replace(["Male", "Female"], [0, 1], inplace=True)
# print(df.head(10))

workclassDummies = pd.get_dummies(df.Workclass)
maritalStatusDummies = pd.get_dummies(df.Marital_Status)
occupationDummies = pd.get_dummies(df.Occupation)
raceDummies = pd.get_dummies(df.Race)
nativeCountryDummies = pd.get_dummies(df.Native_Country)
relationshipDummies = pd.get_dummies(df.Relationship)

df = pd.concat(
    [
        workclassDummies,
        maritalStatusDummies,
        occupationDummies,
        raceDummies,
        nativeCountryDummies,
        relationshipDummies,
        df,
    ],
    axis="columns",
)
df.drop(
    [
        "Workclass",
        "Marital_Status",
        "Occupation",
        "Race",
        "Sex",
        "Native_Country",
        "Education",
        "Relationship",
    ],
    axis=1,
    inplace=True,
)
# print(df.head(10))
# print(df.columns)


X = df.iloc[:, 0:86].values
y = df.iloc[:, 86].values

# print(X[:10])
# print(y[:10])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocess data to fit standard scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply PCA function
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fit Logic Regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting test set result
y_pred = classifier.predict(X_test)

# Make confusion matrix
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
print("True Positive:", TP)
print("False Positive:", FP)
print("True Negative:", TN)
print("False Negative:", FN)

accuracy = (TP + TN) / (TP + FP + TN + FN)
print("Accuracy of the binary classifier = {:0.3f}".format(accuracy))

# Finding performance of various binary classifiers
models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression

models["Logistic Regression"] = LogisticRegression()

# Support Vector Machines
from sklearn.svm import LinearSVC

models["Support Vector Machines"] = LinearSVC()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier

models["Decision Trees"] = DecisionTreeClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier

models["Random Forest"] = RandomForestClassifier()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

models["Naive Bayes"] = GaussianNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

models["K-Nearest Neighbor"] = KNeighborsClassifier()

from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy, precision, recall = {}, {}, {}

for key in models.keys():

    # Fit the classifier
    models[key].fit(X_train, y_train)

    # Make predictions
    predictions = models[key].predict(X_test)

    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)

df_model = pd.DataFrame(
    index=models.keys(), columns=["Accuracy", "Precision", "Recall"]
)
df_model["Accuracy"] = accuracy.values()
df_model["Precision"] = precision.values()
df_model["Recall"] = recall.values()
print(df_model)

ax = df_model.plot.barh()
ax.legend(
    ncol=len(models.keys()), bbox_to_anchor=(0, 1), loc="lower left", prop={"size": 14}
)
plt.tight_layout()
plt.show()
