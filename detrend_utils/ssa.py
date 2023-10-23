import numpy as np
from sklearn.utils.extmath import randomized_svd

class SSA:

    def __init__(self, series, N, K):
        self.series = series
        self.N = N  
        self.K = K

    def transform(self, X):
        L = self.N - self.K + 1
        # 创建大小为L x K的轨迹矩阵
        X = np.zeros((L, self.K))
        X = self.trajectory_matrix(X, self.K) 
        return X

    def decompose(self, X):
        # 奇异值分解
        U, s, VT = np.linalg.svd(X)
        # X_2d = X.reshape(-1, X.shape[-1]) # 尝试用二维数组
        # U, s, VT = randomized_svd(X_2d, self.K, n_iter=5, random_state=None)   
        # 确定秩
        r = np.linalg.matrix_rank(X)
        # 分解
        components = []
        for i in range(r): # 这里分解的对角秩在个数上等于K而已
            comp = s[i] * U[:,i][:,None] @ VT[i,:][:,None].T
            components.append(comp)
        return components

    def reconstruct(self, components, n_components):
        RCs = np.zeros((n_components, (components.shape[1]+self.K-1)))
        for i in range(n_components):
            c = components[i]
            RC = self.average_anti_diag(c)
            RCs[i] = RC
        reconstructed = RCs.sum(axis=0)
        return reconstructed

    @staticmethod
    def trajectory_matrix(X, window_size):

        N = len(X)
        L = N - window_size + 1
        out = []
        
        for i in range(L):
            out.append(X[i:window_size+i])
        
        out = np.asarray(out)
            
        return out

    @staticmethod    
    def average_anti_diag(X):
    
        # 计算反对角线均值的代码 
        
        X = np.asarray(X)
        anti_diag_means = [np.mean(X[::-1, :].diagonal(i)) for i in range(-X.shape[0]+1, X.shape[1])]
        
        return np.asarray(anti_diag_means)