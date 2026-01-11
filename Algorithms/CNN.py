# Code adapted from  https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import os

from Algorithms.utill.data_standar import denormalize_coords, mean_euclidean_distance

from Algorithms.utill.compute_size import calc_out_size
class CNN(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """Constructor for CNN model.

    Params:
    kernel_size (int): the size of CNN kernel
    stride (int): the stride for the CNN
    padding (int): the amount of padding we want on the input
    """
    def __init__(self, kernel_size, stride, padding, dropout):
        super(CNN, self).__init__()
        self.dropout = nn.Dropout(dropout) # Dropout probability

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 32,
                kernel_size = kernel_size,
                # stride = stride,
                stride = stride,
                # padding = padding
                padding = padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,128, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2,)
        )
        h_final, w_final = calc_out_size(23, 23, [
            {"type": "conv", "kernel": kernel_size, "stride": stride, "pad": padding},
            {"type": "pool", "kernel": 2, "stride": 2},
            {"type": "conv", "kernel": kernel_size, "stride": stride, "pad": padding},
            {"type": "pool", "kernel": 2, "stride": 2}
        ])
        self.size = [h_final, w_final]

    """ The forward step of the model.
    Params:
    x (array): the data we are passing to the CNN model

    Returns:
    output (array): the result of running the data through the CNN model
    """
    def forward(self, x):
        # print("Flattened size:", x.shape)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        # flattern the output of conv2
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output
    
    """ Fits the training data and test data to the model (should give preprocessed data).

    Params:
    X_train (array): training samples
    Y_train (array): training labels
    X_test (array): test samples
    Y_test (array): test labels
    """
    def fit(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        # self.standard_scaler = Co_standard_scaler
    
    """ Generates data loader for CNN training phase.

    Params:
    batch_size (int): the size of the batches
    """
    def create_loaders(self, batch_size):
        # Make Training data and Test data into tensors 
        tensor_xtrain = torch.Tensor(self.X_train)
        tensor_ytrain = torch.Tensor(self.Y_train).type(torch.LongTensor) # Labels have to be in LongTensor form
        tensor_xtest = torch.Tensor(self.X_test)
        tensor_ytest = torch.Tensor(self.Y_test).type(torch.LongTensor) # Labels have to be in LongTensor form
        # Make a Tensor Dataset
        train_data = TensorDataset(tensor_xtrain, tensor_ytrain)
        test_data = TensorDataset(tensor_xtest, tensor_ytest)
        # Make Dataloaders
        traindata_loader = DataLoader(train_data, batch_size = batch_size, num_workers = 1)
        testdata_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 1)
        self.loaders = {'train': traindata_loader, 'test': testdata_loader}
    
    """ Trains the CNN model.

    Params:
    num_epochs (int): the number of epochs we use for training
    batch_size (int): the size of the batches we use for learning
    """
    def train_model(self,num_epochs,batch_size=64,eval_train=False, lr = 0.0005, patience=5, sava_model_name = "test.pth", kernel_size = None, stride = None, dropout = None, weight_decay = None):
        self.create_loaders(batch_size)
        # Train the model
        total_step = len(self.loaders['train'])
        optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay=weight_decay)
        best_acc = 0
        train_acc_end = 0
        early_stop = 0
        best_model_state = None
        epoch_count = 0
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.loaders['train']):
                # retrieves batch data
                b_x = images[:,None,:,:] # Changes shape to (batch_size,1,dimension,dimension)
                b_y = labels
                
                # clear gradients for the training step
                optimizer.zero_grad()

                output = self(b_x)
                loss = self.loss_func(output, b_y)
                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()
                if i == total_step - 1: #Prints loss at end of epoch
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i+1, total_step, loss.item()))
            if eval_train:
                train_acc, _ = self.test(self.loaders['train'])
                test_acc, _ = self.test(self.loaders['test'])
                print(f'Training Accuracy: {train_acc}')
                print(f'Test Accuracy: {test_acc}')
                if test_acc > best_acc:
                    best_acc = test_acc
                    train_acc_end = train_acc
                    early_stop = 0
                    best_model_state = self.state_dict() # 保存当前最好模型
                    print("-> Model improved. Saving ")
                else:
                    early_stop += 1
                    print(f"-> No improvement. Patience: {early_stop}/{patience}")
                if early_stop >= patience:
                    print("Early stopping triggered!")
                    break
            epoch_count = epoch+1
            print('-'*30)
        if best_model_state is not None:
            self.save_model(sava_model_name)
            print(f"parameter sava...epoch:{epoch_count} batch_size{batch_size} lr:{lr} kernel_size{kernel_size} stride:{stride} train_acc:{train_acc_end} test_acc:{best_acc}")
            # 构造一行记录
            log_entry = {
                "epoch": epoch_count,
                "batch_size": batch_size,
                "lr": lr,
                "kernel_size": kernel_size,
                "stride": stride,
                "train_acc": train_acc_end,
                "test_acc": best_acc,
                "weight_decay": weight_decay,
                "dropout": dropout
            }

            log_df = pd.DataFrame([log_entry])

            # 文件路径
            log_file = "training_log.xlsx"

            # 是否已有日志文件
            if os.path.exists(log_file):
                # 读取已有文件
                existing_df = pd.read_excel(log_file)
                combined_df = pd.concat([existing_df, log_df], ignore_index=True)
            else:
                combined_df = log_df

            if "序号" not in combined_df.columns:
                combined_df.insert(0, "序号", range(1, len(combined_df) + 1))
            else:
                # 更新已有的序号列，保持递增
                combined_df["序号"] = range(1, len(combined_df) + 1)

            # 添加“序号”列（从1开始）
            # combined_df.insert(0, "序号", range(1, len(combined_df) + 1))

            # 写入 Excel（覆盖原文件）
            combined_df.to_excel(log_file, index=False)
            print("Best model parameters loaded: " + sava_model_name)

    """ Saves the model state.

    Params:
    path (str): the directory where we want to save the model

    Returns:
    (str): Success message 
    """
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        return 'Successfully saved model!'
    
    """ Loads a saved model state from a given path.

    Params:
    path (str): the directory from where we retrieve the saved model

    Returns:
    (str): Success message 
    """
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        # self.eval()
        super().eval()
        return 'Successfully loaded model!'


class CNNClassifier(CNN):
    """Constructor for CNN classification model.

    Params:
    n_classes (int): the number of classes
    kernel_size (int): the size of CNN kernel
    stride (int): the stride for the CNN
    padding (int): the amount of padding we want on the input
    """
    def __init__(self,n_classes, kernel_size=3, stride=1, padding=1, dropout=0.25):
        super().__init__(kernel_size, stride, padding, dropout)
        self.loss_func = nn.CrossEntropyLoss()

        self.fc1 = nn.Linear(128 * self.size[0]*self.size[1], 64)
        # self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(64, n_classes)



    """
    """
    # def _get_flattened_size(self):
    #     with torch.no_grad():
    #         sample = torch.zeros(1, 1, self.input_height, self.input_width)
    #         out = self.conv1(sample)
    #         out = self.dropout(out)
    #         out = self.conv2(out)
    #         out = self.dropout(out)
    #         return out.view(1, -1).shape[1]  # 返回展平后的大小
    """ Tests the model by evaluating test data.  
    
    Params:
    loader: the data we want to test

    Returns:
    (float): the accuracy of the model (%)
    """
    def test(self,loader):
        predictions = []
        self.eval()
        with torch.no_grad():
            correct = 0
            for images, labels in loader:
                test_output = self(images[:,None,:,:])
                pred_y = torch.max(test_output, 1)[1].data.squeeze().tolist()
                predictions += pred_y
                correct += sum(pred_y[i] == labels[i] for i in range(len(labels)))
        return correct/len(loader.dataset)*100, predictions


class CNNRegressor(CNN):
    """Constructor for CNN regression model.

    Params:
    n_outputs (int): number of labels for a given sample
    kernel_size (int): the size of CNN kernel
    stride (int): the stride for the CNN
    padding (int): the amount of padding we want on the input
    """
    def __init__(self,n_targets=1, kernel_size=5, stride=1, padding=1, dropout=0.25):
        super().__init__(kernel_size, stride, padding, dropout)
        self.loss_func = nn.MSELoss()
        # self.fc1 = nn.Linear(128 * 5 * 5, 64)
        self.fc1 = nn.Linear(128*self.size[0]*self.size[1], 64)
        self.fc2 = nn.Linear(64, n_targets)
        self.n_targets = n_targets
    
    """ Tests the model by evaluating test data.  

    Params:
    loader: the data we want to test

    Returns:
    (func): if n_targets > 1 does multi_test instead
    (float): the Mean Squared Error of the model (MSE)
    predictions (array): the set of predictions
    """
    def test(self,loader):
        if self.n_targets > 1:
            return self.multi_test(loader)
        predictions = []
        targets = []
        self.eval()
        with torch.no_grad():
            total_error = 0
            for images, labels in loader:
                test_output = self(images[:,None,:,:])
                predictions += torch.squeeze(test_output).tolist()
                targets += labels.tolist()
                total_error += sum((labels[i].item() - test_output[i].item())**2 for i in range(len(test_output)))
        predictions_np = np.array(predictions)
        targets_np = np.array(targets)
        return predictions_np, targets_np
    
    """ Tests the model by evaluating test data (for multi-target labels).  

    Params:
    loader: the data we want to test

    Returns:
    (float): the Mean Squared Error of the model (MSE)
    predictions (array): the set of predictions
    """
    def multi_test(self,loader):
        predictions = []
        ground_truths = []
        self.eval()
        with torch.no_grad():
            total_error = 0
            for images, labels in loader:
                test_output = self(images[:,None,:,:])
                predictions += torch.squeeze(test_output).tolist()
                ground_truths += labels.tolist()
                for x in range(len(test_output)):
                    pred = test_output[x]
                    elemwise_loss = 0
                    size = len(pred)
                    for i in range(size):
                        elemwise_loss += (labels[x,i] - pred[i])**2 # calculate element wise loss by doing squared error by each target
                    total_error += elemwise_loss/size # Divide total element wise loss by number of targets
        predictions_np = np.array(predictions)
        targets_np = np.array(ground_truths)
        # print(predictions_np)
        # print(targets_np)
        # 打印欧拉距离信息
        return predictions_np, targets_np


    """ Generates data loader for CNN training phase.

    Params:
    batch_size (int): the size of the batches
    """
    def create_loaders(self, batch_size):
        # Make Training data and Test data into tensors 
        tensor_xtrain = torch.Tensor(self.X_train)
        tensor_ytrain = torch.Tensor(self.Y_train).type(torch.FloatTensor)
        tensor_xtest = torch.Tensor(self.X_test)
        tensor_ytest = torch.Tensor(self.Y_test).type(torch.FloatTensor)
        # Make a Tensor Dataset
        train_data = TensorDataset(tensor_xtrain, tensor_ytrain)
        test_data = TensorDataset(tensor_xtest, tensor_ytest)
        # Make Dataloaders
        traindata_loader = DataLoader(train_data, batch_size = batch_size, num_workers = 1)
        testdata_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 1)
        self.loaders = {'train': traindata_loader, 'test': testdata_loader}
    
    """ Trains the CNN model.

    Params:
    num_epochs (int): the number of epochs we use for training
    batch_size (int): the size of the batches we use for learning
    """
    def train_model(self,num_epochs,batch_size=64,eval_train=False, min_max_dist=None, lr=0.001, patience=5, sava_model_name = "test.pth",kernel_size = None, stride = None, weight_decay=0.01, dropout=None):
        self.create_loaders(batch_size)
        # Train the model
        total_step = len(self.loaders['train'])
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_loss = float('inf')
        early_stop_counter = 0
        epoch_count = 0
        best_model_state = None
        shortest_train = 1000
        shortest_test = 1000
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.loaders['train']):
                # retrieves batch data
                b_x = images[:,None,:,:] # Changes shape to (batch_size,1,dimension,dimension)
                b_y = labels
                
                # clear gradients for the training step
                optimizer.zero_grad()

                output = self(b_x)
                loss = self.loss_func(output, b_y.reshape(len(labels),self.n_targets)) # For MSE we need to pass the loss function correctly
                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()
                if i == total_step - 1: # Prints loss at end of epoch
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i+1, total_step, loss.item()))
            if eval_train:
                train_prediction, train_targets = self.test(self.loaders['train'])
                test_prediction, test_targets = self.test(self.loaders['test'])
                train_prediction_denormalize = denormalize_coords(train_prediction, min_max_dist[0], min_max_dist[1])
                train_targets_denormalize = denormalize_coords(train_targets, min_max_dist[0], min_max_dist[1])
                test_prediction_denormalize = denormalize_coords(test_prediction, min_max_dist[2], min_max_dist[3])
                test_targets_denormalize = denormalize_coords(test_targets, min_max_dist[2], min_max_dist[3])
                train_aver_dist = mean_euclidean_distance(train_prediction_denormalize, train_targets_denormalize)
                test_aver_dist = mean_euclidean_distance(test_prediction_denormalize, test_targets_denormalize)
                print(train_aver_dist)
                print(test_aver_dist)
                # Early stopping based on validation metric
                if test_aver_dist < best_loss:
                    best_loss = test_aver_dist
                    early_stop_counter = 0
                    shortest_train = train_aver_dist
                    shortest_test = test_aver_dist
                    best_model_state = self.state_dict()  # 保存当前最好模型
                    print("→ Model improved. Saving...")
                else:
                    early_stop_counter += 1
                    print(f"→ No improvement. Patience: {early_stop_counter}/{patience}")

                if early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break
                # return train_prediction, train_targets, test_prediction, test_targets
            epoch_count = epoch + 1
            print('-'*30)
        if best_model_state is not None:
            # self.load_state_dict(best_model_state)
            self.save_model(sava_model_name)
            print(
                f"parameter sava...epoch:{epoch_count} batch_size{batch_size} lr:{lr} kernel_size{kernel_size} stride:{stride} train_acc:{shortest_train} test_acc:{shortest_test}")
            # 构造一行记录
            log_entry = {
                "epoch": epoch_count,
                "batch_size": batch_size,
                "lr": lr,
                "kernel_size": kernel_size,
                "stride": stride,
                "train_acc": shortest_train,
                "test_acc": shortest_test,
                "weight_decay": weight_decay,
                "dropout": dropout
            }

            log_df = pd.DataFrame([log_entry])

            # 文件路径
            log_file = "training_log.xlsx"

            # 是否已有日志文件
            if os.path.exists(log_file):
                # 读取已有文件
                existing_df = pd.read_excel(log_file)
                combined_df = pd.concat([existing_df, log_df], ignore_index=True)
            else:
                combined_df = log_df

            if "序号" not in combined_df.columns:
                combined_df.insert(0, "序号", range(1, len(combined_df) + 1))
            else:
                # 更新已有的序号列，保持递增
                combined_df["序号"] = range(1, len(combined_df) + 1)

            # 添加“序号”列（从1开始）
            # combined_df.insert(0, "序号", range(1, len(combined_df) + 1))

            # 写入 Excel（覆盖原文件）
            combined_df.to_excel(log_file, index=False)
            print("Best model parameters loaded: " + sava_model_name)
            print("Best model parameters loaded."+ sava_model_name)