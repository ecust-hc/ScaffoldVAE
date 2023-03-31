import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
# import  h5py
# 生成100000*100000的随机矩阵
np.random.seed(42)
A = np.random.rand(1000, 1000)
# with h5py.File('h5file_com4_all.h5', 'r') as hf:
#     A = np.array(hf['elem'])
# 将A划分成多个子矩阵，每个子矩阵大小为1000*1000
block_size = 100
num_blocks = 10
blocks = [A[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
          for i in range(num_blocks) for j in range(num_blocks)]

# 对每个子矩阵计算拉普拉斯矩阵和特征向量
laplacians = []
for block in blocks:
    D = np.diag(np.sum(block, axis=1))
    laplacian = D - block
    _, eigenvectors = eigsh(laplacian, k=20, which='SM')

    laplacians.append(eigenvectors)

# 将所有子矩阵的特征向量拼接成一个大矩阵
eigenvectors = np.concatenate(laplacians, axis=1)

# 对拼接后的特征向量进行KMeans聚类
kmeans = KMeans(n_clusters=5).fit(eigenvectors)

# 输出聚类结果
print(kmeans.labels_)

