import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_file(dsname):
    if dsname == "har": # UCI: DOI:10.24432/C54S4K, Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
        X = np.load("./data/HAR_data.npy", allow_pickle=True)
        y = np.load("./data/HAR_label.npy", allow_pickle=True)
        y = y.reshape(1, len(y))[0]
        return X, y
    elif dsname == "stream_densired": # own
        X = np.load("./data/stream_densired_data.npy", allow_pickle=True)
        y = np.load("./data/stream_densired_label.npy", allow_pickle=True)
        y = y.reshape(1, len(y))[0]
        return X, y
    elif dsname == "rbf3": # from https://github.com/CIG-UFSCar/DS_Datasets
        X = np.load("./data/rbf3_data.npy", allow_pickle=True)
        y = np.load("./data/rbf3_label.npy", allow_pickle=True)
        y = y.reshape(1, len(y))[0]
        return X, y
    elif dsname == "complex9": # from https://github.com/deric/clustering-benchmark
        X = np.load("./data/complex9_data.npy", allow_pickle=True)
        y = np.load("./data/complex9_label.npy", allow_pickle=True)
        y = y.reshape(1, len(y))[0]
        return X, y
    else:
        raise FileNotFoundError

def load_data(dsname, batch=None, batch_max=None, scale=True):
    X, y = read_file(dsname)
    X = X.astype(np.double)
    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    if batch is not None and batch_max is not None:
        X = X[:batch*batch_max]
        y = y[:batch*batch_max]
    return X, y

if __name__ == "__main__":
    for ds in ["complex9", "stream_densired", "rbf3", "har"]:
         data, label = load_data(ds)
         print(f"{ds}: N: {len(label)}, d: {data.shape[1]}, k: {len(np.unique(label))}")

