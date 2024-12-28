import numpy as np


class DataSplitter:
    def __init__(self, val_size=0.2, random_state=None, shuffle=True):
        self.val_size = val_size
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        
        n_samples = X.shape[0]
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
        
        val_split_idx = int(n_samples * (1 - self.val_size))
        
        X_train = X[:val_split_idx]
        y_train = y[:val_split_idx]
        
        X_val = X[val_split_idx:]
        y_val = y[val_split_idx:]
        
        return X_train, X_val, y_train, y_val
    
    def k_fold_split(self, X, y, k):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_size = n_samples // k
        for i in range(k):
            val_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, val_indices)
            
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            yield X_train, X_val, y_train, y_val

class StratifiedDataSplitter(DataSplitter):
    def split(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        
        n_samples = X.shape[0]
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
        
        val_split_idx = int(n_samples * (1 - self.val_size))
        
        X_train, y_train = [], []
        X_val, y_val = [], []
        
        for cls in unique_classes:
            class_size = class_counts[cls]
            val_size_cls = int(class_size * self.val_size)
            
            class_indices_cls = class_indices[cls]
            X_cls = X[class_indices_cls]
            y_cls = y[class_indices_cls]
            
            X_train_cls = X_cls[val_size_cls:]
            y_train_cls = y_cls[val_size_cls:]
            
            X_val_cls = X_cls[:val_size_cls]
            y_val_cls = y_cls[:val_size_cls]
            
            X_train.append(X_train_cls)
            y_train.append(y_train_cls)
            
            X_val.append(X_val_cls)
            y_val.append(y_val_cls)
        
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        
        X_val = np.concatenate(X_val)
        y_val = np.concatenate(y_val)
        
        return X_train, X_val, y_train, y_val
    
    
    
    def k_fold_split(self, X, y, k=5):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        n_samples = X.shape[0]

        # Group indices by class
        unique_classes = np.unique(y)
        class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}

        # Shuffle indices within each class
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            for cls in unique_classes:
                np.random.shuffle(class_indices[cls])

        # Determine the number of samples of each class in each fold
        n_splits = k
        fold_indices = [[] for _ in range(n_splits)] #fold_indices = [0, 1, 2, 3]
        for cls in unique_classes:
            indices_cls = class_indices[cls]                              #indices_cls = np.array(2, 3, 9, 1. 30, 4, 5, 6, 7, 8)
            n_samples_cls = len(indices_cls)
            fold_sizes = np.repeat(n_samples_cls // n_splits, n_splits)   #[2, 2, 2, 2, 2]
            fold_sizes[:n_samples_cls % n_splits] += 1                    #[3, 3, 3, 2, 2]

            current_idx = 0
            for f in range(n_splits):
                start, stop = current_idx, current_idx + fold_sizes[f]
                fold_indices[f].extend(indices_cls[start:stop])
                current_idx = stop

        # Yield the train and validation sets for each fold
        for i in range(n_splits):
            val_indices = np.array(fold_indices[i])
            train_indices = np.concatenate([fold_indices[j] for j in range(n_splits) if j != i])
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            yield X_train, X_val, y_train, y_val