"""
奇异谱分析 (SSA) 模块

用于时间序列的奇异谱分解和重构
"""

import numpy as np


class SSA:
    """
    奇异谱分析 (Singular Spectrum Analysis)

    将时间序列分解为趋势、周期和噪声分量的非参数方法

    Attributes:
        series: 输入时间序列
        n_samples: 序列长度
        window_size: 嵌入维度 (轨迹矩阵窗口大小)
    """

    def __init__(self, series: np.ndarray, window_size: int):
        """
        Args:
            series: 输入时间序列 (1D数组)
            window_size: 嵌入维度 K (轨迹矩阵的列数)
        """
        self.series = np.asarray(series).flatten()
        self.n_samples = len(self.series)
        self.window_size = window_size
        
        if window_size > self.n_samples // 2:
            raise ValueError("窗口大小不应超过序列长度的一半")

    def build_trajectory_matrix(self) -> np.ndarray:
        """
        构建轨迹矩阵 (嵌入)

        Returns:
            轨迹矩阵，形状为 (L, K)，其中 L = N - K + 1
        """
        L = self.n_samples - self.window_size + 1
        trajectory = np.zeros((L, self.window_size))
        
        for i in range(L):
            trajectory[i] = self.series[i:i + self.window_size]
        
        return trajectory

    def decompose(self, trajectory: np.ndarray) -> list[np.ndarray]:
        """
        对轨迹矩阵进行奇异值分解

        Args:
            trajectory: 轨迹矩阵

        Returns:
            奇异分量列表 (每个分量是一个秩1矩阵)
        """
        U, s, Vt = np.linalg.svd(trajectory, full_matrices=False)
        rank = np.linalg.matrix_rank(trajectory)
        
        components = []
        for i in range(rank):
            component = s[i] * np.outer(U[:, i], Vt[i, :])
            components.append(component)
        
        return components

    def reconstruct(self, components: list[np.ndarray], n_components: int) -> np.ndarray:
        """
        通过反反对角平均重构时间序列

        Args:
            components: 奇异分量列表
            n_components: 用于重构的分量数量

        Returns:
            重构后的时间序列
        """
        components = components[:n_components]
        
        # 计算每个分量的重构序列
        reconstructed_components = []
        for comp in components:
            rc = self._average_anti_diagonal(comp)
            reconstructed_components.append(rc)
        
        # 叠加各分量
        return np.sum(reconstructed_components, axis=0)

    @staticmethod
    def _average_anti_diagonal(matrix: np.ndarray) -> np.ndarray:
        """
        计算反对角线平均 (Hankel化)

        Args:
            matrix: 输入矩阵

        Returns:
            反对角线平均后的一维数组
        """
        matrix = np.asarray(matrix)
        rows, cols = matrix.shape
        result_len = rows + cols - 1
        result = np.zeros(result_len)
        
        # 对每个反对角线求平均
        for k in range(result_len):
            # 反对角线索引: (i, j) 满足 i + j = k
            diagonal = []
            for i in range(max(0, k - cols + 1), min(rows, k + 1)):
                j = k - i
                if 0 <= j < cols:
                    diagonal.append(matrix[i, j])
            result[k] = np.mean(diagonal) if diagonal else 0
        
        return result
