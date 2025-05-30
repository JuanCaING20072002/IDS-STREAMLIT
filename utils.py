 # In utils.py
def del_columns(df, columns):
    return df.drop(columns=columns, axis=1)