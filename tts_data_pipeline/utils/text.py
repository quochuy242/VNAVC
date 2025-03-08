import string


def remove_punctuations(sentence: str):
    translator = str.maketrans("", "", string.punctuation)
    return sentence.translate(translator)
