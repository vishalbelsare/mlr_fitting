from scipy.sparse import csr_matrix, coo_matrix
from typing import List, Tuple, Callable, TypedDict, List, Set, Optional, Union 

from scipy.sparse.linalg import splu, LinearOperator, lsmr


from mlrfit.utils import *
from mlrfit.mlr_symm_hpar_matmul import *


def diag_sparseBCt(B, C, rows_lk, ranks):
    # B_compressed: m x (r-1); hpart is symmetric partition
    # return diag(\tilde B \tilde C^T) without permutation
    res = np.zeros(B.shape[0])
    assert B.shape[0] == C.shape[0]
    for level in range(len(rows_lk)):
        num_blocks = rows_lk[level].size - 1 
        for block in range(num_blocks):
            r1, r2 = rows_lk[level][block], rows_lk[level][block+1]
            res[r1:r2] += np.einsum('ij,ji->i', B[r1:r2, ranks[:level].sum():ranks[:level+1].sum()],
                                                C[r1:r2, ranks[:level].sum():ranks[:level+1].sum()].T)
    return res


def frob_loss(perm_A, B, C, rows_lk, cols_lk, ranks):
    # compute loss \|perm_A - \tilde B \tilde C^T\|_F / \|perm_A\|_F
    if isinstance(perm_A, np.ndarray):
        perm_hat_A = compute_perm_hat_A_jit(B, C, rows_lk, cols_lk, ranks)
        return rel_diff(perm_hat_A, den=perm_A)
    elif isinstance(perm_A, tuple):
        assert list(rows_lk) == list(cols_lk), print("A in factorized form is supported only for PSD A")
        A_B, A_C = perm_A 
        n, p = A_B.shape
        if rows_lk[0].size == 2 and rows_lk[0][1] == n:
            ranks0 = ranks + 0
            ranks0[0] += p
            rows_lk_prod = rows_lk
        else:
            ranks0 = np.concatenate([np.array([p]), ranks], axis=0)
            rows_lk_prod = np.concatenate([np.array([0, n]), rows_lk], axis=0)
        B0 = np.concatenate([A_B, -B], axis=1)
        C0 = np.concatenate([A_C,  C], axis=1)
        B_prod, C_prod, ranks_prod = mlr_mlr_symm_hpar_matmul(B0, C0, ranks0, C0, B0, ranks0, rows_lk_prod)
        return np.sqrt(diag_sparseBCt(B_prod, C_prod, rows_lk_prod, ranks_prod).sum()) / np.sqrt(np.einsum('ij,ji->i', A_B.T @ A_B, A_C.T @ A_C).sum())
    

def convert_compressed_to_sparse(B:np.ndarray, hp_entry:EntryHpartDict, \
                        ranks:np.ndarray, mtype='csc', skip_level=np.inf):
    data, i_idx, j_idx = [], [], []
    col_count = 0
    num_levels = len(hp_entry['lk']) 
    for level in range(num_levels):
        if level == skip_level: continue 
        num_blocks = len(hp_entry['lk'][level])-1
        for block in range(num_blocks):
            r1, r2 = hp_entry['lk'][level][block], hp_entry['lk'][level][block+1]
            data += [B[:,ranks[:level].sum():ranks[:level+1].sum()][r1:r2].flatten(order='C')]
            i_idx += [np.tile(np.arange(r1, r2), [ranks[level],1]).flatten(order='F')]
            j_idx += [np.tile(np.arange(col_count, col_count+ranks[level]), [r2-r1])]
            col_count += ranks[level]
    data = np.concatenate(data, axis=0)
    i_idx = np.concatenate(i_idx, axis=0)
    j_idx = np.concatenate(j_idx, axis=0)

    s = sum([(len(hp_entry['lk'][level])-1)*ranks[level] for level in range(num_levels)])
    tilde_B = coo_matrix((data, (i_idx, j_idx)), shape=(B.shape[0], s))
    if mtype == 'csc':
        tilde_B = tilde_B.tocsc()
    elif mtype == 'csr':
        tilde_B = tilde_B.tocsr()
    return tilde_B


class LinOpResidualMatrix:
    """
    Class for residual matrix in LinearOperator form
    """
    def __init__(self, perm_A, B, C, cur_level, rows_lk, cols_lk, ranks):
        A_B, A_C = perm_A 
        m, p = A_B.shape
        n = A_C.shape[0]
        shift = 0
        self.shape = np.array([m, n])
        if cur_level == 0: 
            if rows_lk[0].size == 2 and rows_lk[0][1] == m:
                ranks0 = ranks + 0
                ranks0[0] = p
                shift = ranks[0]
                rows_lk0 = rows_lk; cols_lk0 = cols_lk
                skip_level = np.inf
            else:
                ranks0 = np.concatenate([np.array([p]), ranks], axis=0)
                rows_lk0 = np.concatenate([np.array([0, m]), rows_lk], axis=0)
                cols_lk0 = np.concatenate([np.array([0, n]), cols_lk], axis=0)
                skip_level = 1
        else:
            if rows_lk[0].size == 2 and rows_lk[0][1] == m:
                ranks0 = ranks + 0
                ranks0[0] += p 
                rows_lk0 = rows_lk; cols_lk0 = cols_lk
                skip_level = cur_level
            else:
                ranks0 = np.concatenate([np.array([p]), ranks], axis=0)
                rows_lk0 = np.concatenate([np.array([0, m]), rows_lk], axis=0)
                cols_lk0 = np.concatenate([np.array([0, n]), cols_lk], axis=0)
                skip_level = cur_level + 1

        B0 = np.concatenate([A_B, -B[:, shift:]], axis=1)
        C0 = np.concatenate([A_C,  C[:, shift:]], axis=1)
        self.tilde_B = convert_compressed_to_sparse(B0, {"lk":rows_lk0}, ranks0, skip_level=skip_level, mtype='csr')
        self.tilde_Bt = (self.tilde_B.T).tocsc()
        self.tilde_C = convert_compressed_to_sparse(C0, {"lk":cols_lk0}, ranks0, skip_level=skip_level, mtype='csr')
        self.tilde_Ct = (self.tilde_C.T).tocsc()

    def matvec_slice(self, r1, r2, c1, c2):
        return lambda X: self.tilde_B[r1:r2].dot(self.tilde_Ct[:, c1:c2].dot(X))
    
    def rmatvec_slice(self, r1, r2, c1, c2):
        return lambda X: self.tilde_C[c1:c2].dot(self.tilde_Bt[:, r1:r2].dot(X))
    
    def toarray(self):
        return (self.tilde_B.dot(self.tilde_Ct)).toarray()
    
    def todense(self, r1, r2, c1, c2):
        return (self.tilde_B[r1:r2].dot(self.tilde_Ct[:, c1:c2])).toarray()
    
    def __getitem__(self, key):
        """
        Allows matrix indexing using R[r1:r2, c1:c2] syntax.
        `key` is expected to be a tuple of slices: (row_slice, col_slice)
        """
        if isinstance(key, tuple) and len(key) == 2:
            row_slice, col_slice = key
            r1, r2 = row_slice.start, row_slice.stop
            c1, c2 = col_slice.start, col_slice.stop 
            return LinearOperator((r2-r1, c2-c1), matvec=self.matvec_slice(r1, r2, c1, c2), 
                                  rmatvec=self.rmatvec_slice(r1, r2, c1, c2))
        else:
            raise KeyError("Invalid indexing. Use matrix[row_slice, col_slice]")


def compute_perm_residual(perm_A:Union[np.ndarray, Tuple[np.ndarray,np.ndarray]], B:np.ndarray, C:np.ndarray, cur_level:int, \
                              rows_lk,  cols_lk, ranks:np.ndarray):
    """
    Compute permuted residual for a given level cur_level
    """
    if isinstance(perm_A, np.ndarray):
        return compute_perm_residual_jit(perm_A, B, C, cur_level, rows_lk,  cols_lk, ranks)
    elif isinstance(perm_A, tuple):
        return LinOpResidualMatrix(perm_A, B, C, cur_level, rows_lk,  cols_lk, ranks)
    