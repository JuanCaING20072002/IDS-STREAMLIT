
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd


elimina = ['number__tcp_flags_cwr', 'number__tcp_flags_ecn', 
           'number__tcp_flags_urg', 'number__tcp_flags_ack',
           'number__tcp_flags_push', 'number__tcp_flags_reset',
           'number__tcp_flags_fin', 'number__icmp_type',
           'number__ip_flags_rb', 'number__tcp_flags_res']

class del_columns(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns=columns
#         print('delect columns,columns')
        
        
    def fit(self, X , y=None):
#         print('fit')
#         print(type(X))
        return self
        
    def transform(self, X , y=None):
#         print('trans')
        X =pd.DataFrame(X, columns=self.columns)
#         print(X.head(5))



        X_ =X.drop(elimina,axis=1)
#         print(X_.columns)
#         X_= X_.to_numpy()

        
        return X_