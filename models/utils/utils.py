def custom_encode(series):
    unique_values = series.unique()
    return {val: i for i, val in enumerate(unique_values)}

def encode_dataframe(df, encodings):
    for col, encoding in encodings.items():
        df[col] = df[col].map(encoding)
    return df
