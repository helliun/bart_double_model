#requirements: torch, pandas, scikit-learn
#imports
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

class BARTModel:
    def __init__(self):
        pass

    def concat_model_weights(self, model1, model2):
        """
        This function takes in two BART models and concatenates their weights.

        Input:
        model1: BART model 1
        model2: BART model 2

        Output:
        concatenated_model: BART model with concatenated weights from model 1 and model 2
        """
        new_state_dict = {}
        model1_state_dict = model1.state_dict()
        model2_state_dict = model2.state_dict()

        for k in model1_state_dict.keys():
            if k == "shared.weight":
                new_state_dict[k] = model1_state_dict[k]
            elif k.startswith("decoder") or k.startswith("classification"):
                new_state_dict[k] = model2_state_dict[k]
            else:
                new_state_dict[k] = model1_state_dict[k]

        concatenated_model = model1
        concatenated_model.load_state_dict(new_state_dict)

        return concatenated_model

    def separate_models(self, concatenated_model):
        """
        This function takes in a concatenated BART model and separates it into the original two models.

        Input:
        concatenated_model: BART concatenated model with weights from 2 models

        Output:
        model1: BART model 1 with weights from input concatenated model
        model2: BART model 2 with weights from input concatenated model
        """
        model1_state_dict = {}
        model2_state_dict = {}
        concatenated_state_dict = concatenated_model.state_dict()

        for k in concatenated_state_dict.keys():
            if k == "shared.weight":
                model1_state_dict[k] = concatenated_state_dict[k]
            elif k.startswith("decoder") or k.startswith("classification"):
                model2_state_dict[k] = concatenated_state_dict[k]
            else:
                model1_state_dict[k] = concatenated_state_dict[k]

        model1 = torch.hub.load('pytorch/fairseq', 'bart.large', tokenizer='moses', bpe='fast', force_reload=True)
        model1.load_state_dict(model1_state_dict)

        model2 = torch.hub.load('pytorch/fairseq', 'bart.large', tokenizer='moses', bpe='fast', force_reload=True)
        model2.load_state_dict(model2_state_dict)

        return model1, model2

    def train_concatenated_model(self, model, train_loader, valid_loader, optimizer, criterion, n_epochs):
        """
        This function trains the concatenated model for a specified number of epochs.

        Inputs:
        model: concatenated BART model
        train_loader: pytorch DataLoader object containing training data
        valid_loader: pytorch DataLoader object containing validation data
        optimizer: pytorch optimizer object
        criterion: pytorch loss function object
        n_epochs: number of epochs for training

        Output:
        model: trained concatenated BART model
        train_losses: list of training losses for each epoch
        valid_losses: list of validation losses for each epoch
        """
        train_losses = []
        valid_losses = []

        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
            valid_loss = 0.0

            # train the model
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            # evaluate on validation data
            model.eval()
            with torch.no_grad():
                for batch in valid_loader:
                    inputs, targets = batch
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    valid_loss += loss.item() * inputs.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)

            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {valid_loss}")
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

        return model, train_losses, valid_losses

    def test_concatenated_model(self, model, test_loader):
        """
        This function tests the concatenated model on a dataset and returns evaluation metrics.

        Inputs:
        model: concatenated BART model
        test_loader: pytorch DataLoader object containing test data

        Output:
        accuracy: accuracy score of the model on the test dataset
        precision: weighted precision score of the model on the test dataset
        recall: weighted recall score of the model on the test dataset
        """
        outputs_list = []
        targets_list = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                outputs = model(inputs)
                outputs_list.append(torch.argmax(outputs, dim=1))
                targets_list.append(targets)

        outputs_arr = torch.cat(outputs_list).numpy()
        targets_arr = torch.cat(targets_list).numpy()

        accuracy = accuracy_score(targets_arr, outputs_arr)
        precision = precision_score(targets_arr, outputs_arr, average='weighted')
        recall = recall_score(targets_arr, outputs_arr, average='weighted')

        return accuracy, precision, recall

    def train_separate_models(self, model1, model2, train_loader, valid_loader, optimizer, criterion, n_epochs):
        """
        This function trains two BART models for a specified number of epochs and returns the trained model weights.

        Inputs:
        model1: BART model 1
        model2: BART model 2
        train_loader: pytorch DataLoader object containing training data
        valid_loader: pytorch DataLoader object containing validation data
        optimizer: pytorch optimizer object
        criterion: pytorch loss function object
        n_epochs: number of epochs for training

        Output:
        trained_model1_weights: weights of the first trained model
        trained_model2_weights: weights of the second trained model
        """
        trained_model1_weights = None
        trained_model2_weights = None

        for epoch in range(n_epochs):
            model1.train()
            model2.train()
            train_loss1 = 0.0
            train_loss2 = 0.0

            # train the model1
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model1(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss1 += loss.item() * inputs.size(0)

            # train the model2
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model2(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss2 += loss.item() * inputs.size(0)

            # evaluate on validation data
            model1.eval()
            model2.eval()
            with torch.no_grad():
                valid_loss1 = 0.0
                valid_loss2 = 0.0
                for batch in valid_loader:
                    inputs, targets = batch
                    outputs1 = model1(inputs)
                    loss1 = criterion(outputs1, targets)
                    valid_loss1 += loss1.item() * inputs.size(0)

                    outputs2 = model2(inputs)
                    loss2 = criterion(outputs2, targets)
                    valid_loss2 += loss2.item() * inputs.size(0)

                valid_loss1 = valid_loss1 / len(valid_loader.dataset)
                valid_loss2 = valid_loss2 / len(valid_loader.dataset)

            train_loss1 = train_loss1 / len(train_loader.dataset)
            train_loss2 = train_loss2 / len(train_loader.dataset)

            print(f"Epoch {epoch+1}, Train Loss 1: {train_loss1}, Train Loss 2: {train_loss2}, Val Loss 1: {valid_loss1}, Val Loss 2: {valid_loss2}")
            if train_loss1 < 0.5 and train_loss2 < 0.5:
              trained_model1_weights = model1.state_dict()
              trained_model2_weights = model2.state_dict()
              break
            
        return trained_model1_weights, trained_model2_weights
#changed