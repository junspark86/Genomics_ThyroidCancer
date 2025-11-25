import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def readGenelist(file_path):
    file_path = file_path
    gene_list = []
    with open(file_path, 'r') as f:
        for gene in f:
            gene_list.append(gene.strip())
    return gene_list


def extractFeature(data, nor=False):
    sc = StandardScaler()
    traindata = pd.read_csv('data/1_Thyroid__17901genes_545cases_normalization.csv', index_col=0)
    traindata_nor = sc.fit_transform(traindata.drop('Node',axis=1))
    
    try :
        data = data[traindata.drop('Node',axis=1).columns]
    except Exception as e:
        print('학습에 필요한 gene 발현량이 테스트 데이터에 없습니다.', e)

    # normalization
    if nor: 
        data_nor = sc.transform(data)
    else:
        data_nor = data.values
    data_nor = pd.DataFrame(data=data_nor, index=data.index, columns=data.columns)
    # PCA
    pca = PCA(n_components=2)
    principal_traindata = pca.fit_transform(traindata_nor)
    principal_data = pca.transform(data_nor)
    principal_data = sc.fit_transform(principal_data)
    principal_data = pd.DataFrame(data=principal_data, index=data.index, columns=['PC1', 'PC2'])
    result = pd.concat([data_nor, principal_data], axis=1)
    
    # LDA
    lda = LDA(n_components=1)
    traindata_lda = lda.fit_transform(traindata_nor, traindata['Node'])
    lda_data = lda.transform(data_nor)
    result['lda'] = lda_data
    
    return result