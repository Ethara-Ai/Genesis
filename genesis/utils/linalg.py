import quadrants as qd

import genesis as gs


@qd.func
def mat_mul(A, B, res, n, m, l, i_b):
    """
    Performs matrix multiplication between matrices A and B and stores the result in res.

    Args:
        A (qd.field): The first matrix of shape (n, m, B).
        B (qd.field): The second matrix of shape (m, l, B).
        res (qd.field): The result matrix of shape (n, l, B).
        n (int): The number of rows in matrix A and res.
        m (int): The number of columns in matrix A and rows in matrix B.
        l (int): The number of columns in matrix B and res.
        i_b (int): batch index.
    """
    pass


@qd.func
def mat_mul_vec(mat, vec, res, n, m, i_b):
    pass


@qd.func
def mat_inverse(mat, L, U, y, res, n, i_b):
    """
    Inverse via LU decomposition
    """
    pass


@qd.func
def mat_add(A, B, n, m, i_b):
    pass


@qd.func
def mat_transpose(A, B, n, m, i_b):
    pass


@qd.func
def mat_add_eye(A, x, n, i_b):
    pass


@qd.func
def mat_mask(A, mask, n, m, i_b):
    pass
