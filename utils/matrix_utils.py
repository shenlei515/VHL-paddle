
import numpy as np
import paddle


# def orthogo_tensor(x):
#     from sympy.matrices import Matrix, GramSchmidt
#     m, n = x.size()
#     x_np = x.t().numpy()
#     matrix = [Matrix(col) for col in x_np.T]
#     gram = GramSchmidt(matrix)
#     ort_list = []
#     for i in range(m):
#         vector = []
#         for j in range(n):
#             vector.append(float(gram[i][j]))
#         ort_list.append(vector)
#     ort_list = np.mat(ort_list)
#     ort_list = paddle.from_numpy(ort_list)
#     ort_list = F.normalize(ort_list,dim=1)
#     return ort_list



def orthogo_tensor(raw_dim, column_dim):
    a = paddle.rand([raw_dim, column_dim])
    u, s, v = paddle.linalg.svd(a)
    return u
























