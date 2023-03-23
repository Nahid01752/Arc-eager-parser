import numpy as np


class FeatureMapping:
    """  extracting features and adding to the feature dictionary """
    def __init__(self):
        self.feature_map = {}
        self.new_mapping = 1  # index for unseen feature
        self.frozen = False  # freeze feature mapping

    # extract features and update feature map
    def templates(self, state, sentence):
        feats = np.empty(0, int)
        stack = state.stack
        buffer = state.buffer
        form = sentence.forms
        pos = sentence.pos
        lemma = sentence.lemmas
        morph = sentence.morphs
        head = state.heads
        left_most = state.left_most
        right_most = state.right_most
        s0 = stack[-1]  # top of the stack
        b0 = buffer[0]   # front of the buffer
        distance = str(b0 - s0)     # distance b2n top of the stack and front of the buffer

        # features for stack elements
        feats = np.append(feats, form[s0])
        feats = np.append(feats, pos[s0])
        feats = np.append(feats, lemma[s0])
        feats = np.append(feats, form[s0] + pos[s0])
        feats = np.append(feats, lemma[s0] + pos[s0])
        feats = np.append(feats, form[s0] + lemma[s0])

        if head[s0] >= 0:
            feats = np.append(feats, form[head[s0]])
            feats = np.append(feats, pos[head[s0]])
        if left_most[s0] >= 0:
            feats = np.append(feats, form[left_most[s0]])
            feats = np.append(feats, pos[left_most[s0]])
        if right_most[s0] >= 0:
            feats = np.append(feats, form[right_most[s0]])
            feats = np.append(feats, pos[right_most[s0]])

        if len(stack) > 1:
            s1 = stack[-2]      # second top of the stack
            feats = np.append(feats, form[s1])
            feats = np.append(feats, pos[s1])
            feats = np.append(feats, form[s1] + pos[s1])
            if morph:
                # adding german morphs
                feats = np.append(feats, morph[s0])

        # features for buffer elements
        feats = np.append(feats, form[b0])
        feats = np.append(feats, pos[b0])

        if head[b0] >= 0:
            feats = np.append(feats, form[head[b0]])
            feats = np.append(feats, pos[head[b0]])
        if left_most[b0] >= 0:
            feats = np.append(feats, form[left_most[b0]])
            feats = np.append(feats, pos[left_most[b0]])
        if right_most[b0] >= 0:
            feats = np.append(feats, form[right_most[b0]])
            feats = np.append(feats, pos[right_most[b0]])

        if len(buffer) > 1:
            b1 = buffer[1]
            feats = np.append(feats, form[b1])
            feats = np.append(feats, pos[b1])
            feats = np.append(feats, form[b1] + pos[b1])
            if stack:
                feats = np.append(feats, pos[b0] + pos[b1] + pos[stack[-1]])
                
        if len(buffer) > 2:
            b2 = buffer[2]
            feats = np.append(feats, form[b2])
            feats = np.append(feats, pos[b2])
            feats = np.append(feats, form[b2] + pos[b2])

        if morph:
            feats = np.append(feats, morph[b0])
            feats = np.append(feats, pos[b0] + morph[b0])

        # features for both buffer and stack elements
        feats = np.append(feats, form[s0] + pos[s0] + form[b0] + pos[b0])
        feats = np.append(feats, form[s0] + pos[s0] + form[b0])
        feats = np.append(feats, form[s0] + pos[s0] + pos[b0])
        feats = np.append(feats, form[s0] + form[b0] + pos[b0])
        feats = np.append(feats, pos[s0] + form[b0] + pos[b0])
        feats = np.append(feats, pos[s0] + pos[b0])

        if morph:
            feats = np.append(feats, morph[s0] + morph[b0])
            feats = np.append(feats, pos[s0] + morph[s0] + pos[b0] + morph[b0])

        if head[s0] >= 0:
            feats = np.append(feats, pos[s0] + pos[b0] + pos[head[s0]])
        if left_most[b0] >= 0:
            feats = np.append(feats, pos[s0] + pos[b0] + pos[left_most[s0]])
            feats = np.append(feats, pos[s0] + pos[b0] + pos[left_most[b0]])
        if right_most[b0] >= 0:
            feats = np.append(feats, pos[s0] + pos[b0] + pos[right_most[s0]])
            feats = np.append(feats, pos[s0] + pos[b0] + pos[right_most[b0]])

        # feature: apply distance
        feats = np.append(feats, form[s0] + distance)
        feats = np.append(feats, pos[s0] + distance)
        feats = np.append(feats, form[b0] + distance)
        feats = np.append(feats, pos[b0] + distance)
        feats = np.append(feats, form[s0] + form[b0] + distance)
        feats = np.append(feats, pos[s0] + pos[b0] + distance)
        feats = np.append(feats, lemma[s0] + lemma[b0] + distance)

        feats_vector = []
        for feat in feats:
            if self.frozen:
                if feat in self.feature_map:
                    feats_vector.append(self.feature_map[feat])
            else:
                # add feature mapping if new
                update = self.update_map(feat)
                feats_vector.append(update)

        return feats_vector

    def update_map(self, feature):
        if feature not in self.feature_map:
            self.feature_map[feature] = self.new_mapping  # add new mapping
            self.new_mapping += 1
        return self.feature_map[feature]
