from sklearn.preprocessing import StandardScaler
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc, roc_curve
from scipy.interpolate import BSpline

app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
app.config['UPLOAD_FOLDER'] = 'app/static'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

def prepare_data_train(fname):
	""" read and prepare training data """
	data = pd.read_csv(fname)
	clean = data.drop(['id' ], axis = 1)
	return clean

def prepare_labels_train(fname):
	labels = pd.read_csv(fname)
	clean_labels = labels.drop(['id' ], axis = 1)
	return clean_labels

def get_batch(dataset,target, batch_size=2000, val=False, index=None):
    num_features = 32
    window_size = 1024
    if val == False:
        index = random.randint(window_size, len(dataset) - 16 * batch_size)
        indexes = np.arange(index, index + 16*batch_size, 16)

    else:
        indexes = np.arange(index, index + batch_size)
    
    batch = np.zeros((batch_size, num_features, window_size//4))
    
    b = 0
    for i in indexes:
        
        start = i - window_size if i - window_size > 0 else 0
        
        tmp = dataset[start:i]
        batch[b,:,:] = tmp[::4].transpose()
        
        b += 1

    targets = target[indexes]
    return torch.DoubleTensor(batch), torch.DoubleTensor(targets) 

class convmodel(nn.Module):
  def __init__(self, drop=0.5):
    super().__init__()
    self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=0, stride=1)
    self.bn = nn.BatchNorm1d(64)
    self.pool = nn.MaxPool1d(2, stride=2)
    self.dropout1 = nn.Dropout(drop)
    self.conv = nn.Sequential(self.conv2, nn.ReLU(inplace=True), self.bn,self.pool, self.dropout1)
  def forward(self, x):
    x = self.conv(x)
    return x

class Combine(nn.Module):
    def __init__(self,out_classes):
        super(Combine, self).__init__()
        self.cnn = convmodel().double()
        self.rnn = nn.LSTM(input_size=127, hidden_size=64, num_layers=1,batch_first=True)
        self.linear = nn.Linear(64,out_classes)

    def forward(self, x):
      x = self.cnn(x)
      out, hidden=self.rnn(x)
      out = self.linear(out[:, -1, :])
      return torch.sigmoid(out)

def valscore(preds, targs):
    aucs = [auc(targs[:, j], preds[:, j]) for j in range(6)]
    total_loss = np.mean(aucs)
    return total_loss

@app.route('/')
def home():
    return render_template('index.html', title='EEG Classifier')

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        files = request.files.getlist('file[]')
        for file in files:
        	filename = secure_filename(file.filename)
        	fp = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        	file.save(fp)
        
        raw = []
        yraw = []
        
        subjects= [1]
        series = [1,2,3,4]
        

        for subject in subjects:
        	for ser in series:
        		 datafname ='subj%d_series%d_data.csv' % (subject,ser)
        		 eventsfname ='subj%d_series%d_events.csv' % (subject,ser)

        		 fullpath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], datafname)
        		 eventsfullpath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], eventsfname)

        		 data = prepare_data_train(fullpath)
        		 labels = prepare_labels_train(eventsfullpath)
        		 raw.append(data)
        		 yraw.append(labels)

        X = pd.concat(raw)
        Y = pd.concat(yraw) 

        X = np.asarray(X.astype(float))
        Y = np.asarray(Y.astype(float))

        scaler = StandardScaler().fit(X)
        data = scaler.transform(X)

        path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], 'cnnlstm.pt')
    	
        model = Combine(6).double()
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        model.eval()
        
        batchsize = 2000
        i = 1024
        p = []
        res = []

        while i < len(data):
            if i + batchsize > len(data):
                batchsize = len(data) - i
            x, y = get_batch(data, Y, batchsize, index=i, val=True)
            x = (x)
            preds = model(x)
            preds = preds.squeeze(1)
            p.append(np.array(preds.cpu().data))
            res.append(np.array(y.data))
            i += batchsize
        preds = p[0]
        for i in p[1:]:
            preds = np.vstack((preds,i))
        targs = res[0]
        for i in res[1:]:
            targs = np.vstack((targs, i))
        auc_score = valscore(preds, targs)

    return render_template("success.html", name = str(auc_score))  



if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0')