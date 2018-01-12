import spacy
import os
import collections


def feature_function(training_texts, answer_texts, features, labels):

    new_labels = labels

    incidents = {"arson": '', "attack": '', "bombing": '', "kidnapping": '', "robbery": ''}

    for index in range(len(training_texts)):
        answers = parse_answer(answer_texts[index], labels)

        for label in labels:
            if label in incidents:
                if label.upper() in answers["incident"]:
                    new_labels[label] += "+1 "
                else:
                    new_labels[label] += "-1 "
                for text in training_texts[index]:
                    text = text.orth_.strip()
                    if text.isalpha():
                        if text in features:
                            new_labels[label] += str(features[text]) + ':1 '
                        else:
                            new_labels[label] += str(features["UNK"]) + ':1 '
                new_labels[label] += '\n'
            else:
                for text in training_texts[index]:
                    text_string = text.orth_.strip()
                    if text_string.isalpha():
                        if text_string in answers[label]:
                            new_labels[label] += "+1 "
                        else:
                            new_labels[label] += "-1 "
                        if text_string in features:
                            new_labels[label] += str(features[text_string]) + ':1 '
                        else:
                            new_labels[label] += str(features["UNK"]) + ':1 '
                        new_labels[label] += str(features["POSTAG" + text.pos_]) + ':1\n'

    return new_labels


def parse_answer(answer, labels):
    answers = {}

    fields = {"incident": '', "weapon": '', "perpindiv": '', "perporg": '', "target": '', "victim": ''}

    for label in fields:
        answers[label] = set()

    for lines in answer.splitlines():
        if not lines:
            continue
        first = lines.split()[0]
        if first == "PERP":
            first += " " + lines.split()[1]
        if first[len(first)-1] == ':':
            curr_label = first
            if curr_label == "ID:":
                continue
            elif curr_label == "INCIDENT:":
                answers["incident"].add(lines.split()[1])
            elif curr_label == "WEAPON:":
                answers["weapon"].add(lines.split()[1])
            elif curr_label == "PERP INDIV:":
                answers["perpindiv"].add(lines.split()[2])
            elif curr_label == "PERP ORG:":
                answers["perporg"].add(lines.split()[2])
            elif curr_label == "TARGET:":
                answers["target"].add(lines.split()[1])
            elif curr_label == "VICTIM:":
                answers["victim"].add(lines.split()[1])
        else:
            if curr_label == "INCIDENT:":
                answers["incident"].add(lines.split()[0])
            elif curr_label == "WEAPON:":
                answers["weapon"].add(lines.split()[0])
            elif curr_label == "PERP INDIV:":
                answers["perpindiv"].add(lines.split()[0])
            elif curr_label == "PERP ORG:":
                answers["perporg"].add(lines.split()[0])
            elif curr_label == "TARGET:":
                answers["target"].add(lines.split()[0])
            elif curr_label == "VICTIM:":
                answers["victim"].add(lines.split()[0])

    return answers


def word_features(word, features):
    string = word.orth_.strip()
    feats = {}
    if string in features:
        feats[features[string]] = 1
    if ("POSTAG" + word.pos_) in features:
        feats[features["POSTAG" + word.pos_]] = 1
    return feats


def story_features(story, features):
    feats = {}
    for token in story:
        string = token.orth_.strip()
        if string in features:
            feats[features[string]] = 1
    return feats


def run():

    # Load spacy
    nlp = spacy.load('en')

    # Training texts will be loaded as spacy objects
    training_texts = []

    # Loaded as a single string
    answer_texts = []

    text_dir = "/Users/matthewcanova/Desktop/NLP/NLP_Project/developset/texts"
    ans_dir = "/Users/matthewcanova/Desktop/NLP/NLP_Project/developset/answers"

    for filename in os.listdir(text_dir):
        with open(text_dir + '/' + filename) as text:
            training_texts.append(nlp(text.read()))
        with open(ans_dir + '/' + filename + ".anskey") as text:
            answer_texts.append(text.read())

    # Dict with all incidents in it
    incidents = {"arson": '', "attack": '', "bombing": '', "kidnapping": '', "robbery": ''}

    pos_tags = {}
    features = {}
    feature_id = 1
    for text in training_texts:
        for token in text:
            token_string = token.orth_.strip()
            token_pos = "POSTAG" + token.pos_
            if token_string.isalpha():
                if token_string not in features:
                    features[token_string] = feature_id
                    feature_id += 1
                if token_pos not in features:
                    features[token_pos] = feature_id
                    feature_id += 1

    features["UNK"] = feature_id

    # All classifiers we wish to get vector data for
    labels = {"arson": '', "attack": '', "bombing": '', "kidnapping": '', "robbery": '', "weapon": '', "perpindiv": '',
              "perporg": '', "target": '', "victim": ''}

    # Returns a dict with (classifier, vector)
    feature_text = feature_function(training_texts, answer_texts, features, labels)

    with open("feature_indices.txt", 'w') as file:
        for feature, index in features.items():
            file.write(feature + ':' + str(index) + ' ')

    # Write training vector data to file
    for label in feature_text:
        with open("training_vectors/" + label + ".txt.vector", 'w') as file:
            file.write(feature_text[label])

