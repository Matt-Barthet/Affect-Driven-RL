import numpy as np
from scipy.stats import mode
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class KNNSurrogateModel:
    def __init__(self, k, classifier, preference, cluster):
        self.x_train = None
        self.y_train = None
        self.k = k
        self.classifier = classifier
        self.preference = preference
        self.cluster = cluster
        self.scaler = MinMaxScaler()
        self.load_data()

    def __call__(self, state):
        distances = np.array(np.sqrt(np.sum((state - self.x_train) ** 2, axis=1)))
        k_indices = np.array(np.argsort(distances)[:self.k])
        k_labels = np.array(self.y_train)[k_indices]
        if self.k == 1:
            return self.y_train[k_indices][0]

        if self.classifier:
            predicted_class = mode(k_labels)
        else:
            weights = 1 / (distances[k_indices] + 1e-5)
            weighted_sum = np.sum(weights * k_labels)
            total_weights = np.sum(weights)
            predicted_class = weighted_sum / total_weights
        return predicted_class, k_indices

    def load_and_clean(self, filename, cluster, preference):
        data = pd.read_csv(filename)
        data = data.loc[:, data.apply(pd.Series.nunique) != 1]
        if preference:
            if cluster != 0:
                data = data[data['Cluster_R'] == cluster]
            data = data[data['Ranking'] != "stable"]
            arousals = data['Ranking'].values
            data = data.drop(columns=['Player', 'Ranking'])
            label_mapping = {"decrease": 0.0, "increase": 1.0}
            arousals = [label_mapping[label] for label in arousals]
        else:
            if cluster != 0:
                data = data[data['Cluster'] == cluster]
            arousals = data['[output]arousal'].values
            data = data.drop(columns=['[control]player_id', '[output]arousal', 'botRespawn'])
        data = data[data.columns[~data.columns.str.startswith("Cluster")]]
        data = data[data.columns[~data.columns.str.startswith("Time_Index")]]
        data = data[data.columns[~data.columns.str.contains("arousal")]]
        data = data[data.columns[~data.columns.str.contains("botRespawn")]]
        data = data[data.columns[~data.columns.str.contains("Score")]]
        return data, arousals

    def load_data(self):
        unscaled_data, _ = self.load_and_clean(f'./Datasets/Solid_3000ms_nonorm_with_clusters.csv', self.cluster, False)
        if self.preference:
            self.x_train, self.y_train = self.load_and_clean(f'./Datasets/solid_3000ms_1W_downsampled_pair_data.csv', self.cluster, self.preference)
        else:
            self.x_train, self.y_train = self.load_and_clean(f'./Datasets/Solid_3000ms_minmax_with_clusters.csv', self.cluster, self.preference)
        self.scaler.fit(unscaled_data.values)
