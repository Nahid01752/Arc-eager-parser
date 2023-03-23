import numpy as np


class Parser:
    """Parser class that checks preconditions, decodes saved model, calculates scores, and predicts transitions"""

    def __init__(self, state, sentence, feature_map):
        self.state = state
        self.sentence = sentence
        self.feature_map = feature_map
        self.data = []

    def decode(self, parsing_model):
        weights = parsing_model.weights
        while self.state.buffer:
            feature_list = self.feature_map.templates(self.state, self.sentence)
            scores = np.zeros((4,))
            for idx in feature_list:
                scores += weights[:, idx]  # scores for current transition

            predicted = np.argsort(-scores)  # highest transition in decreasing order
            for item in predicted:
                if item == 0 and self.can_left_arc():
                    self.state.left_arc()
                    break
                elif item == 1:
                    self.state.right_arc()
                    break
                elif item == 2 and self.can_reduce():
                    self.state.reduce()
                    break
                elif item == 3:
                    self.state.shift()
                    break

        return self.state.heads

    # preconditions
    def can_left_arc(self):
        stack_top = self.state.stack[-1]
        return stack_top != 0 and self.state.heads[stack_top] == -1

    def can_reduce(self):
        return self.state.heads[self.state.stack[-1]] != -1

    def can_shift(self):
        return len(self.state.buffer) >= 1 or self.state.stack

