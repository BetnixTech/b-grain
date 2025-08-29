# ml_pipeline_framework.py

import csv, json, random, math, re
from typing import List, Any, Dict, Union, Tuple
from collections import Counter
from PIL import Image
import numpy as np

# --- Base Transformer ---
class Transformer:
    def fit(self, X: List[List[Any]], y: List[Any]=None):
        return self
    def transform(self, X: List[List[Any]]) -> List[List[Any]]:
        return X
    def fit_transform(self, X: List[List[Any]], y: List[Any]=None) -> List[List[Any]]:
        return self.fit(X,y).transform(X)

# --- Numeric Transformer ---
class NumericNormalizer(Transformer):
    def __init__(self, method='minmax'):
        self.method = method
        self.stats = []

    def fit(self, X, y=None):
        self.stats = []
        for col in zip(*X):
            vals = [float(v) for v in col]
            if self.method=='minmax':
                self.stats.append((min(vals), max(vals)))
            elif self.method=='zscore':
                mean = sum(vals)/len(vals)
                std = math.sqrt(sum((v-mean)**2 for v in vals)/len(vals))
                self.stats.append((mean,std))
        return self

    def transform(self,X):
        new_X=[]
        for row in X:
            new_row=[]
            for val,(a,b) in zip(row,self.stats):
                v=float(val)
                if self.method=='minmax':
                    new_row.append((v-a)/(b-a) if b!=a else 0)
                elif self.method=='zscore':
                    new_row.append((v-a)/b if b!=0 else 0)
            new_X.append(new_row)
        return new_X

# --- Categorical Transformer ---
class CategoricalEncoder(Transformer):
    def fit(self,X,y=None):
        self.encoders=[]
        for col in zip(*X):
            unique = sorted(set(col))
            self.encoders.append({v:i for i,v in enumerate(unique)})
        return self
    def transform(self,X):
        new_X=[]
        for row in X:
            new_row=[]
            for val, enc in zip(row,self.encoders):
                new_row.append(enc.get(val,0))
            new_X.append(new_row)
        return new_X

# --- Text Transformer ---
class TextVectorizer(Transformer):
    def fit(self,X,y=None):
        self.vocabs=[]
        for col in zip(*X):
            vocab={}
            for val in col:
                for word in re.findall(r'\w+', str(val).lower()):
                    if word not in vocab: vocab[word]=len(vocab)
            self.vocabs.append(vocab)
        return self
    def transform(self,X):
        new_X=[]
        for row in X:
            new_row=[]
            for val,vocab in zip(row,self.vocabs):
                vec=[0]*len(vocab)
                for w in re.findall(r'\w+', str(val).lower()):
                    if w in vocab:
                        vec[vocab[w]]=1
                new_row.extend(vec)
            new_X.append(new_row)
        return new_X

# --- Image Transformer ---
class ImageTransformer(Transformer):
    def __init__(self,size=(32,32)):
        self.size=size
    def transform(self,X):
        new_X=[]
        for row in X:
            new_row=[]
            for val in row:
                if isinstance(val,str):
                    img = Image.open(val).resize(self.size)
                    arr = np.array(img).flatten()
                    new_row.extend(arr.tolist())
                else:
                    new_row.append(val)
            new_X.append(new_row)
        return new_X

# --- Pipeline ---
class Pipeline:
    def __init__(self, steps: List[Tuple[str, Transformer]]):
        self.steps = steps
    def fit(self,X,y=None):
        for _,step in self.steps:
            X=step.fit_transform(X,y)
        return self
    def transform(self,X):
        for _,step in self.steps:
            X=step.transform(X)
        return X
    def fit_transform(self,X,y=None):
        for _,step in self.steps:
            X=step.fit_transform(X,y)
        return X

# --- Dataset Utilities ---
def shuffle_split(X,y,train_ratio=0.8):
    combined=list(zip(X,y))
    random.shuffle(combined)
    X,y=zip(*combined)
    n=int(len(X)*train_ratio)
    return list(X[:n]), list(y[:n]), list(X[n:]), list(y[n:])
def batch_generator(X,y,batch_size=32):
    for i in range(0,len(X),batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

# --- Example Usage ---
if __name__=="__main__":
    # Sample CSV
    import io
    csv_file = io.StringIO("num,cat,text,label\n1,A,Hello,0\n2,B,World,1\n3,A,Hello,0")
    with open("sample.csv","w") as f: f.write(csv_file.getvalue())

    # Load CSV
    X=[]
    y=[]
    with open("sample.csv") as f:
        reader=csv.DictReader(f)
        for row in reader:
            X.append([row['num'],row['cat'],row['text']])
            y.append(row['label'])

    # Build pipeline
    pipeline = Pipeline([
        ('numeric', NumericNormalizer()),
        ('categorical', CategoricalEncoder()),
        ('text', TextVectorizer())
    ])

    X_processed = pipeline.fit_transform(X)
    X_train,y_train,X_test,y_test = shuffle_split(X_processed,y)
    print("Train:",X_train,y_train)
    print("Test:",X_test,y_test)

    # Batch example
    for Xb,yb in batch_generator(X_train,y_train,batch_size=2):
        print("Batch:",Xb,yb)
