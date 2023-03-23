import numpy as np


class Instance:
    """ Instance saves the transition label with its corresponding feature vectors """
    def __init__(self, label, feats_vector):
        self.label = label
        self.feats_vector = feats_vector


class Oracle:
    """ Oracle checks the optimal conditions before executing transition """
    def __init__(self, state, sentence, feature_map):
        self.state = state
        self.sentence = sentence
        self.map = feature_map
        self.data = []

    # predict valid transitions
    def oracle_rules(self):
        train_data = []
        while self.state.buffer:
            feats_vector = np.asarray(self.map.templates(self.state, self.sentence))
            if self.state.stack and self.should_left_arc():
                self.state.left_arc()
                instance = Instance(0, feats_vector)
            elif self.state.stack and self.should_right_arc():
                self.state.right_arc()
                instance = Instance(1, feats_vector)
            elif self.state.stack and self.should_reduce():
                self.state.reduce()
                instance = Instance(2, feats_vector)
            else:
                self.should_shift()
                instance = Instance(3, feats_vector)
            train_data.append(instance)
        self.data = train_data
        return self.data

    # optimal conditions

    def should_left_arc(self):
        return True if (self.state.buffer[0], self.state.stack[-1]) in self.sentence.gold_arcs else False

    def should_right_arc(self):
        return True if (self.state.stack[-1], self.state.buffer[0]) in self.sentence.gold_arcs else False

    def has_all_children(self, stack_top):
        return all(self.state.heads[dep] == stack_top for head, dep in self.sentence.gold_arcs if head == stack_top)

    def should_reduce(self):
        return self.has_head(self.state.stack[-1]) and self.has_all_children(self.state.stack[-1])

    def has_head(self, stack_top):
        return self.state.heads[stack_top] != -1

    def should_shift(self):
        last_buffer = self.state.buffer.popleft()
        self.state.stack.append(last_buffer)
