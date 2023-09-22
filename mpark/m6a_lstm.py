import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, MaxPooling1D, Conv1D, Flatten
from tensorflow.keras.metrics import AUC


class M6ALstm(Sequential):
    def __init__(
            self, num_hidden_layers=1, num_units=32, dropout_rate=0.1,
            activation_function='relu', learning_rate=0.05, momentum_value=0.7,
            batch_size=32, num_nucleotides=7
    ):
        super().__init__()
        
        self.num_nucleotides = num_nucleotides
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum_value = momentum_value

        full_window = num_nucleotides * 2 + 1
        input_shape = ((full_window), 5)

        for _ in range(num_hidden_layers):
            if _ == 0:
                self.add(LSTM(num_units, input_shape=input_shape, return_sequences=(num_hidden_layers > 1)))
            else:
                self.add(LSTM(num_units, return_sequences=(_ < num_hidden_layers)))

            self.add(Dropout(dropout_rate))

        self.add(Dense(1, activation='sigmoid'))
    
    @staticmethod
    def filter_set_by_base_quality(features, labels, quality_threshold=0.05, drop_base_quality=True):

        list_above_threshold = []
        
        for i in range(features.shape[0]):
            if features[i, 4, 7] > quality_threshold:
                list_above_threshold.append(i)
        
        above_threshold_indices = np.array(list_above_threshold)
        
        above_threshold_features = features[above_threshold_indices]
        above_threshold_labels = labels[above_threshold_indices]
        
        if drop_base_quality:
            above_threshold_features = above_threshold_features[:, np.arange(above_threshold_features.shape[1]) != 4, :]
        
        return above_threshold_features, above_threshold_labels

    @staticmethod
    def set_num_nucleotides(features, num = 7):
        if num > 7:
            print('Cannot have more than 7 nucleotides.')
        else:
            num_to_remove = 7 - num
            
            keep_from_start = num_to_remove
            keep_from_end = features.shape[2] - num_to_remove

            new_features = features[:, :, keep_from_start:keep_from_end]
            
            return new_features
    
    def fit_semisupervised(
            self,
            train_features=None,
            train_labels=None,
            val_features=None,
            val_labels=None,
            max_epochs=10,
            device="cpu",
            best_save_model="",
            final_save_model="",
            prev_aupr=0,
    ):
        # filter train set
        train_features, y_train = self.filter_set_by_base_quality(train_features, train_labels)
        train_features = self.set_num_nucleotides(train_features, self.num_nucleotides)
        X_train = train_features.transpose((0, 2, 1))

        # filter val set    
        val_features, y_val = self.filter_set_by_base_quality(val_features, val_labels)
        val_features = self.set_num_nucleotides(val_features, self.num_nucleotides)
        X_val = val_features.transpose((0, 2, 1)) 
        print(X_val.shape, y_val.shape)

        optimizer = Adam(learning_rate=self.learning_rate, beta_1=self.momentum_value)
        self.compile(
            loss='binary_crossentropy', 
            optimizer=optimizer, 
            metrics=['accuracy', 'precision', 'recall', AUC(curve='PR')]  
            # Average precision is the area under the PR curve
        )
        for epoch in range(max_epochs):
            print('before hist')
            history = self.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, batch_size=self.batch_size)
            print('after hist')
            val_accuracy = history.history['val_accuracy'][-1]
            print(
                f"Epoch {epoch + 1}/{max_epochs}, " 
                f"Val Acc: {val_accuracy:.4f}")