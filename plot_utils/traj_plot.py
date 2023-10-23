import matplotlib.pyplot as plt

def plot_trajectory_matrix(X, max_rows=None, max_cols=None):
        
        plt.imshow(X[:max_rows, :max_cols], cmap='coolwarm')
        plt.xlabel('Time Step')
        plt.ylabel('Trajectory Index')
        plt.title('Trajectory Matrix')
        plt.show()