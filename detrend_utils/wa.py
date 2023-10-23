import numpy as np

class WeightedAverage:
    """
    @example:
    weights = np.array([[0.4], [0.15], [0.15], [0.15], [0.15]])
    wa = WeightedAverage(weights)
    car_40_per_wa_trend = wa.rolling(CAR, window=5, min_periods=1, center=True)
    """

    def __init__(self, weights):
        self.weights = weights

    def rolling(self, data, window=5, min_periods=1, center=True):
        
        weighted_data = []
        
        # for i in range(len(data)):
        #     start = i - window + 1
        #     end = i + 1
        #     window_data = data[start:end]
        #     if len(window_data) >= min_periods:
        #         weighted_window = np.dot(window_data.T, self.weights[:len(window_data)])
        #         weighted_data.append(weighted_window)
        # return weighted_data
        return data.rolling(window=window, min_periods=min_periods, center=center).apply(lambda x: np.dot(x, self.weights[:len(x)]))