# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def least_squares_GD(y, tx, initial_w,max_iters, gamma):
        # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return ws,losses


def least_squares_SGD(y, tx, initial_w,max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return ws,losses

def least_squares(y, tx):
    """calculate the least squares."""
    w = np.linalg.solve((tx.T).dot(tx),(tx.T).dot(y))
    loss = calculate_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = y.shape[0]
    txTtx = tx.T.dot(tx)
    w = np.linalg.solve((tx.T).dot(tx) + (lambda_*np.identity(tx.shape[1])),(tx.T).dot(y))
    loss = calculate_mse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):

    assert y.shape[0] == tx.shape[0], "#rows of y and tx must be equal"
    assert tx.shape[1] == initial_w.shape[0], "#col of tx must be equal to the #rows of initial_w"
    assert max_iters >= 0, "max_iters must be non-negative"

    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w, fn="sig")
        w -= gamma * grad

    loss = compute_loss(y, tx, w, fn="sig")
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    assert y.shape[0] == tx.shape[0], "#rows of y and tx must be equal"
    assert tx.shape[1] == initial_w.shape[0], "#col of tx must be equal to the #rows of initial_w"
    assert max_iters >= 0, "max_iters must be non-negative"

    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w, fn="sig") + 2 * lambda_ * w
        w -= gamma * grad

    loss = compute_loss(y, tx, w, fn="sig") + lambda_ * (w.T @ w) / 2
    return w, loss



def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
