import numpy as np
from scipy.stats import mode
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from scipy import stats


def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)  # Ensure the input data is a numpy array
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)  # Margin of error
    return np.round(mean, 4), np.round(ci, 4)


class KNNSurrogateModel:
    def __init__(self, k, classifier, preference, cluster, game):
        self.x_train = None
        self.y_train = None
        self.k = k
        self.classifier = classifier
        self.preference = preference
        self.cluster = cluster
        self.scaler = MinMaxScaler()
        self.game = game

        self.max_score = 0
        if game == "fps":
            self.max_score = 500
        elif game == "pirates":
            self.max_score = 460
        else:
            self.max_score = 24

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
            label_mapping = {"decrease": 0.0, "increase": 1.0}
            arousals = [label_mapping[label] for label in arousals]
            data['Arousal'] = arousals

            participant_list = data['Player'].unique()
            human_arousal = []
            for participant in participant_list:
                sub_df = data[data['Player'] == participant]
                max_score = np.mean(sub_df['Arousal'])
                human_arousal.append(max_score)
            print("Human Arousal: ", compute_confidence_interval(human_arousal))

            data = data.drop(columns=['Player', 'Ranking', 'Arousal'])

        else:
            if cluster != 0:
                data = data[data['Cluster'] == cluster]

            arousals = data['[output]arousal'].values

            participant_list = data['[control]player_id'].unique()
            human_arousal = []
            for participant in participant_list:
                sub_df = data[data['[control]player_id'] == participant]
                max_score = np.max(sub_df['playerScore'])
                human_arousal.append(max_score / self.max_score)  # Keep normalized score
            print("Human Scores: ", compute_confidence_interval(human_arousal))

            data = data.drop(columns=['[control]player_id', '[output]arousal'])

        if self.game == "Solid":
            data = data[data.columns[~data.columns.str.contains("botRespawn")]]

        data = data[data.columns[~data.columns.str.startswith("Cluster")]]
        data = data[data.columns[~data.columns.str.startswith("Time_Index")]]
        data = data[data.columns[~data.columns.str.contains("arousal")]]

        if self.game != "fps":
            data = data[data.columns[~data.columns.str.contains("Score")]]
        # print(data.columns)
        return data, arousals

    def load_data(self):
        unscaled_data, _ = self.load_and_clean(f'./Datasets/{self.game}_3000ms_nonorm_with_clusters.csv', self.cluster, False)
        if self.preference:
            self.x_train, self.y_train = self.load_and_clean(f'./Datasets/{self.game}_3000ms_pairs_classification_downsampled.csv', self.cluster, self.preference)
        else:
            self.x_train, self.y_train = self.load_and_clean(f'./Datasets/{self.game}_3000ms_minmax_with_clusters.csv', self.cluster, self.preference)
        self.scaler.fit(unscaled_data.values)
