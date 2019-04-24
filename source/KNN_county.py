import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
class KNN_county:
    def __init__(self, adj_list, degree_sep):
        self.mapping = {}
        self.adj_list = adj_list
        self.degree_sep = degree_sep

    def fit(self, X_train, y_train):
        self.mapping = {int(X_train[i]):int(y_train[i]) for i in range(len(y_train))}


    def get_all_neighbors(self, fips):
        sep = self.degree_sep 
        neighbors = []
        seen = set()
        curr_fipsQ = [fips]
        while sep > 0 and curr_fipsQ: 
            curr_fips = curr_fipsQ.pop()
            try:
                curr_neighbors = self.adj_list.get(int(curr_fips))
            except:
                raise Exception("what the fuckkk is", curr_fips)
            sep -= 1
            if not curr_neighbors:
                continue
            else:
                for n in list(curr_neighbors):
                    if n not in seen:
                        seen.add(n)
                        neighbors.append(n)
                        curr_fipsQ.append(n)
        return neighbors


    def predict_one(self, fips):
        neighbors = self.get_all_neighbors(fips)
        # print(len(neighbors))
        if not neighbors:
            return 0
        neighbor_labels = []
        for n in neighbors:
            label = self.mapping.get(n)
            # print(label)
            if label:
                neighbor_labels.append(label)
        if neighbor_labels:
            labels, counts = np.unique(neighbor_labels, return_counts = True)
            return labels[np.argmax(counts)]
        else:
            return 0


    def predict(self, X_test):
        y_pred = []
        for fips in X_test:
            y_pred.append(self.predict_one(fips))
        return y_pred

    def mse_score(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    def num_mistakes(self, y_true, y_pred):
        return np.sum(np.array(y_true) != np.array(y_pred))









class Baseline:
    def __init__(self):
        pass 

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        return np.random.randint(4, size = len(X_test))

    def mse_score(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    def num_mistakes(self, y_true, y_pred):
        return np.sum(np.array(y_true) != np.array(y_pred))
