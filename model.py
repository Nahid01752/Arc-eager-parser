"""
transition based arc-eager parser with perceptron
"""
import dill
import numpy as np


class Perceptron:
    def __init__(self, feature_map, weights):
        self.map = feature_map
        self.weights = weights

    def save_model(self, model, ln):
        stream = open('model_5_' + ln, 'wb')
        dill.dump(model, stream, -1)
        stream.close()

    def update_weights(self, data, predicted):
        # Update weights based on the difference between predicted and actual labels
        for idx in data.feats_vector:
            self.weights[data.label][idx] += 1  # add 1 to the correct prediction
            self.weights[predicted][idx] -= 1  # subtract 1 from the wrong prediction

    def my_model(self, train_data):
        epochs = 5
        for i in range(epochs):
            total = 0
            correct = 0
            print("epoch: ", i + 1)

            for data in train_data:
                total += 1
                scores = np.zeros((4,))

                # Compute scores for each possible label
                for index in data.feats_vector:
                    for j in range(0, 4):
                        scores[j] += self.weights[j][index]

                predicted = np.argmax(scores)    # Get the label with the highest score

                # Update weights if predicted label is different from actual label
                if predicted != data.label:
                    self.update_weights(data, predicted)

                # Keep track of number of correct predictions
                if predicted == data.label:
                    correct += 1

                # Print training accuracy every 500 instances
                if total % 500 == 0:
                    accuracy = 100.0 * correct / total
                    print("total:", total, "accuracy:", accuracy)


# file_path = 'english/train/wsj_train.only-projective.conll06'  # train file location
# file_path = 'english/train/wsj_train.only-projective.first-1k.conll06'
# file_path = 'german/train/tiger-2.2.train.only-projective.first-1k.conll06'
# file_path = 'german/train/tiger-2.2.train.only-projective.conll06'
# language = 'de'
# train(file_path, language)
