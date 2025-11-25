import pandas as pd
import numpy as np
from utils_tabnet import readGenelist, extractFeature
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test():
    # data prepare
    X_test =  data.values
    
    # model define
    model = TabNetClassifier()
    
    # load weight
    model.load_model('weights/tabnet_f_classif_val0.87_test0.87.zip')
    
    # test
    y_predprob = model.predict_proba(X_test)
    y_predvalue = model.predict(X_test)
    y_pred = np.hstack((y_predprob[:,-1].reshape(-1, 1), y_predvalue.reshape(-1, 1)))
    # saved result
    result = pd.DataFrame(y_pred, index=data.index, columns=['predict proba', 'predict value'])
    result['predict value'] = result['predict value'].apply(lambda x : 'N1' if x else 'N0')
    result.to_csv(result_path)
    

    
if __name__ == "__main__":
    # set parameters
    data_path = 'data/1_Thyroid__17901genes_545cases_normalization.csv'
    gene_list_path = 'data/gene_f_classif_corr_top9.txt'
    result_path = 'result.csv'
    
    batch_size = 20
    epochs = 500
    LR = 0.01
    n_features = 9
    
    
    # preprocessing data

    data = pd.read_csv(data_path, index_col=0)
    data = extractFeature(data, nor=True)
    gene_list = readGenelist(gene_list_path)
    data= data[gene_list]
    
    test()