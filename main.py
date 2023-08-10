import torch
from torch import nn
import matplotlib.pyplot as plt
"""
1. Data Preparation
2. Build a model
3. Fitting the model to data (training)
4. Making predictions and evaluating the model (inference)
5. Saving and loading the model
6. Putting it all together
"""

"""
1. Data Preparation
"""
# create known parameters
weight = 0.7
bias = 0.3

# create dummy data
start = 0
end = 1
step = 0.1

X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

# create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Visualize the data in graph
def plot_prediction(train_data=X_train,train_label=y_train,test_data=X_test,test_label=y_test,prediction=None):
    """
    Plots training data, test data and compare predictions
    :param train_data:
    :param train_label:
    :param test_data:
    :param test_label:
    :param prediction:
    :return:
    """
    plt.figure(figsize=(10,7))

    #Plot training data in blue
    plt.scatter(train_data, train_label, c="b", s=4, label="Training Data")

    #Plot tesat data in green
    plt.scatter(test_data,test_label,c="g",s=4, label="Test Data")

    #if prediction is available
    if prediction is not None:
        prediction_np = prediction.detach().numpy()  # Detach and convert to NumPy
        plt.scatter(test_data,prediction_np,c="r",s=4, label = "Predictions")

    # show the legend
    plt.legend(prop={"size": 14})

"""
2. Build a model
"""
# create a linear regression class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float32))

    # Forward method to define the computation in the model
    def forward(self,X:torch.Tensor):
        return self.weight * X + self.bias

# create a random seed
torch.manual_seed(42)

model_0 = LinearRegressionModel()


y_preds = model_0(X_test)

plot_prediction(prediction=y_preds)
plt.show()