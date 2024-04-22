import faiss
import numpy as np

if __name__ == '__main__':
    d = 64                           # dimension
    nb = 10000                     # database size
    nq = 20                         # nb of queries
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    # set HNSW index parameters
    M = 64  # number of connections each vertex will have
    ef_search = 32  # depth of layers explored during search
    ef_construction = 64  # depth of layers explored during index construction

    # initialize index (d == 128)
    index = faiss.IndexHNSWFlat(d, M)
    # set efConstruction and efSearch parameters
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.verbose = True
    # add data to index
    # index.add(X)
    # index.add(nb, xb)
    index.add(xb)

    faiss.write_index(index, "./hnsw.dat")

    # index.

    D, I = index.search(xq, 10)

    print(I)
