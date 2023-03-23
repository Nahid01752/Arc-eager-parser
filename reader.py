class Tokens:
    """ Class to store tokens """
    def __init__(self):
        self.forms = []
        self.lemmas = []
        self.pos = []
        self.morphs = []
        self.heads = []
        self.gold_arcs = []


# this function reads sentences from datasets
def corpus(path, language, blind=False):
    data = []  # list to hold sentences
    with open(path) as file:
        tokens = Tokens()
        for line in file:
            if line.strip():
                val = line.split("\t")
                if val[0] == "1":
                    # add root
                    tokens.forms.append('ROOT')
                    tokens.lemmas.append('ROOT')
                    tokens.pos.append('ROOT_POS')
                    # no heads and gold_arcs for blind set
                    if not blind:
                        tokens.heads.append('_')
                    if language == 'de':
                        tokens.morphs.append('ROOT_MORPH')
                tokens.forms.append(val[1])
                tokens.lemmas.append(val[2])
                tokens.pos.append(val[3])
                if not blind:
                    tokens.heads.append(int(val[6]))
                    tokens.gold_arcs.append((int(val[6]), int(val[0])))
                if language == 'de':
                    tokens.morphs.append(val[5])
            else:
                data.append(tokens)
                tokens = Tokens()
    return data
