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
    # normalization
    if nor and ('Node' in data.columns):
        data_nor = sc.fit_transform(data.drop('Node',axis=1))
    elif nor:
        data_nor = sc.fit_transform(data)
    elif 'Node' in data.columns:
        data_nor = data.drop('Node',axis=1)
    else:
        pass
        
    # PCA
    pca = PCA(n_components=2)
    principal_data = pca.fit_transform(data_nor)
    principal_data = sc.fit_transform(principal_data)
    principal_data = pd.DataFrame(data=principal_data, index=data.index, columns=['PC1', 'PC2'])
    result = pd.concat([data, principal_data], axis=1)
    
    # LDA
    lda = LDA(n_components=1)
    lda_data = lda.fit_transform(data_nor, data['Node'])
    result['lda'] = lda_data
    
    return result