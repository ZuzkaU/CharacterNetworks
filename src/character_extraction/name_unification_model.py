#!/usr/bin/env python3
import argparse
import pickle
import lzma
import os
import logging

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier

model_file = "name_unification.model"

def getModel(model_path):
    with lzma.open(model_path, "rb") as f:
        model = pickle.load(f)
        return model


def trainModel(data_path, out_file=model_file, random_state=13):
    train_data, train_target = getTrainData(data_path)
    
    logging.info("Training the model...")
    model = MLPClassifier(random_state=random_state)
    model.fit(train_data, train_target)
    logging.info("Model trained.")
    
    with lzma.open(out_file, "wb") as f:
        pickle.dump(model, f)
        
    logging.info("Model saved to {}".format(out_file))
    
    return model


def getTrainData(data_path):
    logging.info('Getting training data from {}...'.format(data_path))
    
    books = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            parts = file.split('.')
            if len(parts) > 1 and parts[1] == "weights":
                books.append(os.path.join(root, parts[0]))
        
    train_data = []
    train_target = []
    
    if len(books) == 0:
        raise Exception("No training data found!")
    
    for name in books:
        logging.info(name)
        characters = {}
        with open(name + '.csv') as f:
            for line in f.read().splitlines()[1:]:
                parts = line.split(',')
                characters[parts[1]] = int(parts[0])
        
        with open(name + '.weights') as f:
            for line in f.read().splitlines():
                parts = line.split(',')
                char_A, char_B = parts[0], parts[1]
                weights = [int(n) for n in parts[2:]]
                
                if not char_A in characters or not char_B in characters:
                    continue
                
                if characters[char_A] == characters[char_B]:
                    train_target.append(1)
                else:
                    train_target.append(0)
                train_data.append(weights)
    
    return train_data, train_target

