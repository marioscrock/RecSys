import scipy
import numpy as np
from scipy import sparse as sps
from toRec import reduceKNN
import saveload
from math import log

class ISimilarity(object):
    """Abstract interface for the similarity metrics"""

    def __init__(self, shrinkage=10):
        self.shrinkage = shrinkage

    def compute(self, X):
        pass


class Cosine(ISimilarity):
    def compute(self, X):
        # convert to csc matrix for faster column-wise operations
        X = check_matrix(X, 'csc', dtype=np.float32)

        # 1) normalize the columns in X
        # compute the column-wise norm
        # NOTE: this is slightly inefficient. We must copy X to compute the column norms.
        # A faster solution is to  normalize the matrix inplace with a Cython function.
        Xsq = X.copy()
        Xsq.data **= 2
        norm = np.sqrt(Xsq.sum(axis=0))
        norm = np.asarray(norm).ravel()
        norm += 1e-6
        # compute the number of non-zeros in each column
        # NOTE: this works only if X is instance of sparse.csc_matrix
        col_nnz = np.diff(X.indptr)
        # then normalize the values in each column
        X.data /= np.repeat(norm, col_nnz)
        print("Normalized")

        # 2) compute the cosine similarity using the dot-product
        dist = X * X.T
        print("Computed")

        # zero out diagonal values
        dist = dist - sps.dia_matrix((dist.diagonal()[scipy.newaxis, :], [0]), shape=dist.shape)
        print("Removed diagonal")

        # and apply the shrinkage
        if self.shrinkage > 0:
            dist = self.apply_shrinkage(X, dist)
            print("Applied shrinkage")

        return dist

    def apply_shrinkage(self, X, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind * X_ind.T
        # remove the diagonal
        co_counts = co_counts - sps.dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        co_counts_shrink = co_counts.copy()
        co_counts_shrink.data += self.shrinkage
        co_counts.data /= co_counts_shrink.data
        dist.data *= co_counts.data
        return dist

class AdjustedCosine(ISimilarity):

    def compute(self, X):

        X = check_matrix(X, 'csc', dtype=np.float32)
        col_nnz = np.diff(X.indptr)

        # For each column computes the mean
        means = np.array(np.mean(X, axis=0)).ravel()

        #To each row element removes mean of related column
        X.data -= np.repeat(means, col_nnz)
        print('Means removed')

        Xsq = X.copy()
        Xsq.data **= 2
        norm = np.sqrt(Xsq.sum(axis=0))
        norm = np.asarray(norm).ravel()
        norm += 1e-6

        X.data /= np.repeat(norm, col_nnz)
        print("Normalized")

        dist = X * X.T
        print("Computed")

        dist = dist - sps.dia_matrix((dist.diagonal()[scipy.newaxis, :], [0]), shape=dist.shape)
        print("Removed diagonal")

        # and apply the shrinkage
        if self.shrinkage > 0:
            dist = self.apply_shrinkage(X, dist)
            print("Applied shrinkage")

        return dist

    def compute_diff_matrices(self, X, Y):

        X = check_matrix(X, 'csc', dtype=np.float32)
        Y = check_matrix(Y, 'csc', dtype=np.float32)

        col_nnz_x = np.diff(X.indptr)
        col_nnz_y = np.diff(Y.indptr)

        means_x = np.array(np.mean(X, axis=0)).ravel()
        means_y = np.array(np.mean(Y, axis=0)).ravel()

        X.data -= np.repeat(means_x, col_nnz_x)
        Y.data -= np.repeat(means_y, col_nnz_y)
        print('Means removed')

        Xsq = X.copy()
        Xsq.data **= 2
        norm_x = np.sqrt(Xsq.sum(axis=0))
        norm_x = np.asarray(norm_x).ravel()
        norm_x += 1e-6

        Ysq = Y.copy()
        Ysq.data **= 2
        norm_y = np.sqrt(Ysq.sum(axis=0))
        norm_y = np.asarray(norm_y).ravel()
        norm_y += 1e-6

        X.data /= np.repeat(norm_x, col_nnz_x)
        Y.data /= np.repeat(norm_y, col_nnz_y)
        print("Normalized")

        dist = X * Y.T
        print("Computed")

        return dist.tocsr()

    def apply_shrinkage(self, X, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind * X_ind.T
        # remove the diagonal
        co_counts = co_counts - sps.dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        co_counts_shrink = co_counts.copy()
        co_counts_shrink.data += self.shrinkage
        co_counts.data /= co_counts_shrink.data
        dist.data *= co_counts.data
        return dist

class Pearson(ISimilarity):

    def compute(self, X):

        X = check_matrix(X, 'csr', dtype=np.float32)
        row_nnz = np.diff(X.indptr)

        # For each row computes the mean
        means = np.array(np.mean(X, axis=1)).ravel()

        # To each column element removes mean of related row
        X.data -= np.repeat(means, row_nnz)
        print('Means removed')

        Xsq = X.copy()
        Xsq.data **= 2
        norm = np.sqrt(Xsq.sum(axis=1))
        norm = np.asarray(norm).ravel()
        norm += 1e-6

        X.data /= np.repeat(norm, row_nnz)
        print("Normalized")

        dist = X * X.T
        print("Computed")

        dist = dist - sps.dia_matrix((dist.diagonal()[scipy.newaxis, :], [0]), shape=dist.shape)
        print("Removed diagonal")

        # and apply the shrinkage
        if self.shrinkage > 0:
            dist = self.apply_shrinkage(X, dist)
            print("Applied shrinkage")

        return dist

    def apply_shrinkage(self, X, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind * X_ind.T
        # remove the diagonal
        co_counts = co_counts - sps.dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        co_counts_shrink = co_counts.copy()
        co_counts_shrink.data += self.shrinkage
        co_counts.data /= co_counts_shrink.data
        dist.data *= co_counts.data
        return dist


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)

def delete_rows_lil(mat, i):
    if not isinstance(mat, sps.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - len(i), mat._shape[1])


