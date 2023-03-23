import dill
import numpy as np
from reader import corpus
from state import State
from decoder import Parser
from oracle import Oracle
from features import FeatureMapping
from model import Perceptron


# training data
def train(path, ln):
    # Initialize stack and feature map
    stack = np.arange(1)
    feature_map = FeatureMapping()

    # Load sentences from corpus
    sentences = corpus(path, ln, blind=False)

    # Loop through instances and generate training data
    data_inst = []
    for inst in sentences:
        buffer = np.arange(1, len(inst.forms))
        state = State(stack, buffer)
        oracle = Oracle(state, inst, feature_map)
        parser_data = oracle.oracle_rules()
        data_inst.append(parser_data)

    # Flatten and concatenate the training data
    data_inst = np.concatenate(data_inst).flatten().tolist()

    # Freeze the feature map and initialize weights
    feature_map.frozen = True
    weights = np.zeros((4, feature_map.new_mapping), dtype=np.float32)

    # Train the model using perceptron algorithm
    model = Perceptron(feature_map, weights)
    model.my_model(data_inst)

    # Save the trained model
    model.save_model(model, ln)


def load_model(language):
    """ Load pre-trained model """
    with open(f"model_5_{language}", "rb") as file:
        model = dill.load(file)
    return model


def evaluate(path, language):
    """ Evaluate the model on the dev set """
    sentences = corpus(path, language, blind=False)
    model = load_model(language)
    total, correct = 0, 0
    for inst in sentences:
        stack = [0]  # Initialize the stack
        buffer = list(range(1, len(inst.forms)))  # Initialize the buffer
        state = State(stack, buffer)
        heads = Parser(state, inst, model.map).decode(model)
        for gold_head, curr_head in zip(inst.heads[1:], heads[1:]):
            if curr_head == gold_head:
                correct += 1
            total += 1
    accuracy = (correct / total) * 100
    print(f"Dev Accuracy: {accuracy:.2f}")


def predict(path, language):
    """ Predict heads on the test set """
    sentences = corpus(path, language, blind=True)
    model = load_model(language)
    with open(f"parsed_file_{language}.conll06", "w") as file:
        for inst in sentences:
            stack = [0]  # Initialize the stack
            buffer = list(range(1, len(inst.forms)))  # Initialize the buffer
            state = State(stack, buffer)
            heads = Parser(state, inst, model.map).decode(model)
            for i in range(1, len(heads)):
                if language == "en":
                    line = f"{i}\t{inst.forms[i]}\t{inst.lemmas[i]}\t{inst.pos[i]}\t_\t_\t{heads[i]}\t_\t_\t_"
                elif language == "de":
                    line = f"{i}\t{inst.forms[i]}\t{inst.lemmas[i]}\t{inst.pos[i]}\t_\t{inst.morphs[i]}\t{heads[i]}\t_\t_\t_"
                file.write(line + "\n")
            file.write("\n")
    print(f"Predictions saved to parsed_file_{language}.conll06")


if __name__ == "__main__":
    train_path = "german/train/tiger-2.2.train.only-projective.first-1k.conll06"
    dev_path = "german/dev/tiger-2.2.dev.conll06.gold"
    test_path = "german/test/tiger-2.2.test.conll06.blind"
    language = "de"
    train(train_path, language)
    evaluate(dev_path, language)
    predict(test_path, language)
