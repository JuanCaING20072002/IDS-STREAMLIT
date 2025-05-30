#Proyecto IDS Tesis Fernando Gutierrez P.
import numpy as np
import pandas as pd
from sklearn import metrics
import info_sistema


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def metricas(df,tipo_ataque, Trans_cluster):
    '''
    label: 
    cluster: 
    '''
    print(metrics.classification_report( df[tipo_ataque], df[Trans_cluster] ))
    print('Purity ', round(purity_score(df[tipo_ataque], df[Trans_cluster]),5))
    print('homogeneity_score: ', round(metrics.homogeneity_score(df[tipo_ataque], df[Trans_cluster]),5))
    print('completeness_score: ', round(metrics.completeness_score(df[tipo_ataque], df[Trans_cluster]),5))
    print('v_measure_score: ', round(metrics.v_measure_score(df[tipo_ataque], df[Trans_cluster]),5))
    #print('adjusted_rand_score: ', round(metrics.adjusted_rand_score(y['tipo_ataque_num'], result['predictions']),5))
    print('adjusted_mutual_info_score: ', round(metrics.adjusted_mutual_info_score(df[tipo_ataque], df[Trans_cluster]),5))
    

def metrica_internas(pp3,cluster):
    

    ss = round(metrics.silhouette_score(pp3, cluster,metric='sqeuclidean'),5)
#     ss='prueba_DBSCAN'
    chs = round(metrics.calinski_harabasz_score(pp3, cluster),5)
    dbs = round(metrics.davies_bouldin_score(pp3, cluster),5)
    return (ss,chs,dbs)
#     return (chs,dbs)

    


#Asignar los nombres a los clusters predichos acorde con las categorias conocidas(tipo_ataque)=(y) 
def y(df,start_, n_clusters,cluster,tipo_ataque):
    l=[]
    for ClusterNum in range(start_, n_clusters):

        OneCluster = pd.DataFrame(df[df[cluster] == ClusterNum].groupby(tipo_ataque).size())
        OneCluster.columns=['Size']
    #     print(f'{OneCluster}')
        NewDigit = OneCluster.index[OneCluster['Size'] == OneCluster['Size'].max()].tolist()
        NewDigit[0]
    #     print(f'{NewDigit[0]}')

        rowIndex = df.index[df[cluster] == ClusterNum]
        df.loc[rowIndex, 'Trans_cluster'] = NewDigit[0]

        print(ClusterNum, NewDigit[0])
        l=l+[(ClusterNum, NewDigit[0])]
    return l

# def specifity(df1):
def matriz(df1,tipo_ataque,Trans_cluster):
    
    cm = metrics.confusion_matrix(df1[tipo_ataque], df1[Trans_cluster], labels=df1[tipo_ataque].value_counts().index)
    cm = pd.DataFrame(cm, columns=df1[tipo_ataque].value_counts().index, index=df1[tipo_ataque].value_counts().index)
    tabla_conf = pd.DataFrame(index=df1[tipo_ataque].value_counts().index, columns=['tp','fp','fn','tn'])
#     display matriz confusion
    matriz = pd.DataFrame(index=['Positive','Negative'], columns=['Positive','Negative'])
    for i in range(cm.shape[0]):

        tp=cm.iloc[i,i]
        fp=cm.iloc[:,i].values.sum()-tp
        fn=cm.iloc[i,:].values.sum()-tp
        tn=cm.iloc[:i,:i].values.sum()+cm.iloc[i+1:,:i].values.sum() +cm.iloc[:i,i+1:].values.sum() +cm.iloc[i+1:,i+1:].values.sum()
        matriz.iloc[0,0]=tp
        matriz.iloc[0,1]=fn
        matriz.iloc[1,0]=fp
        matriz.iloc[1,1]=tn
        print('')
        print(df1[tipo_ataque].value_counts().index[i])
        print(matriz)
        tabla_conf.iloc[i,:] = tp,fp,fn,tn

    tabla_conf['tp+fn'] = tabla_conf['tp']+tabla_conf['fn']
    tabla_conf['tp+fp'] = tabla_conf['tp']+tabla_conf['fp']
    tabla_conf['fn+tn'] = tabla_conf['fn']+tabla_conf['tn']
    tabla_conf['fp+tn'] = tabla_conf['fp']+tabla_conf['tn']
#     recall= tp / (tp + fn)
    tabla_conf['recall'] = tabla_conf['tp']/tabla_conf['tp+fn']
#     precision_score= tp / (tp + fp)
    tabla_conf['precision'] = tabla_conf['tp']/tabla_conf['tp+fp']
#     F1 = 2 * (precision * recall) / (precision + recall)
    tabla_conf['F1'] = (2*tabla_conf['precision']*tabla_conf['recall'])/(tabla_conf['precision']+tabla_conf['recall'])
    
    
    tabla_binaria = pd.DataFrame(index=[0,1], columns=['Cantidad','coincidencia','% prediccion','total prediciones'])
    tabla_binaria.iloc[0,0]= tabla_conf.iloc[0,4]
    tabla_binaria.iloc[1,0]=tabla_conf.iloc[0,7]
    tabla_binaria.iloc[0,1]= tabla_conf.iloc[0,0]
    tabla_binaria.iloc[1,1]=tabla_conf.iloc[0,3]
    tabla_binaria.iloc[0,2]=tabla_binaria.iloc[0,1]/tabla_binaria.iloc[0,0]
    tabla_binaria.iloc[1,2]=tabla_binaria.iloc[1,1]/tabla_binaria.iloc[1,0]
    tabla_binaria.iloc[0,3]=tabla_conf.iloc[0,5]
    tabla_binaria.iloc[1,3]=tabla_conf.iloc[0,6]
    return tabla_binaria ,tabla_conf

