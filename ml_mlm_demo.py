import numpy as np
import scipy.io

from sklearn.preprocessing import MinMaxScaler
from src.sklearn_mlmlm.classifiers import MultiLabelMLMClassifier

# load test data
dataname = "SYNTHETIC"
SCALING = True

metrics_str = [
    "ACCURACY",
    "HAMMING_LOSS",
    "MICROF1",
    "MACROF1",
    "RANKING_LOSS",
    "COVERAGE",
    "ONE_ERROR",
    "AVERAGE_PRECISION",
]

print("Loading " + str(dataname) + "data set...")
data = scipy.io.loadmat(
    "./examples/INPUT/" + str(dataname) + "/MAT-FORMAT/" + str(dataname) + ".mat"
)
Xtrain = data["Xtrain"]
Ytrain = data["Ytrain"]
Xtest = data["Xtest"]
Ytest = data["Ytest"]

if SCALING:
    print("Minmax-scaling input data set...")
    scaler = MinMaxScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

N, M = Xtrain.shape
_, L = Ytrain.shape

grid_temp = np.arange(3, 6.05, 0.05)
p_grid = 2**grid_temp


print("Running LOOCV ML-MLM training...")

model = MultiLabelMLMClassifier(p_grid)
model.fit(Xtrain, Ytrain)
ypred = model.predict(Xtest)
yscore = model.predict_proba(Xtest)
print(ypred)
print(yscore)
matches = ypred.astype(int) == Ytest
accuracy = (matches.astype(int).sum()) / (matches.shape[0] * matches.shape[1])
print(accuracy)
