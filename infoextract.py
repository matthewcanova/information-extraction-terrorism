import spacy
import sys
import classifier
import ie_model

# Load spacy
nlp = spacy.load('en')


def label_story(story, labels, features):
    story_output = ''

    # ID:
    story_output += "ID: " + story.splitlines()[0].split()[0] + '\n'
    story = story[1:]
    story = nlp(story)

    # INCIDENT:
    incidents = {"arson": '', "attack": '', "bombing": '', "kidnapping": '', "robbery": ''}
    story_output += "INCIDENT: "

    for inc in incidents:
        word_vec = ie_model.story_features(story, features)
        result = classifier.run_perceptron(word_vec, labels[inc][0], labels[inc][1])
        if result > 0:
            story_output += inc.upper() + '\n'
            break
    else:
        story_output += "ATTACK\n"

    # WEAPON
    story_output += "WEAPON: "
    empty = True
    for word in story:
        word_vec = ie_model.word_features(word, features)
        result = classifier.run_perceptron(word_vec, labels["weapon"][0], labels["weapon"][1])
        if result > 0:
            story_output += word + '\n'
            empty = False
    if empty:
        story_output += "-"
    story_output += '\n'

    # PERP INDIV:
    story_output += "PERP INDIV: "
    empty = True
    for word in story:
        word_vec = ie_model.word_features(word, features)
        result = classifier.run_perceptron(word_vec, labels["perpindiv"][0], labels["perpindiv"][1])
        if result > 0:
            story_output += word + '\n'
            empty = False
    if empty:
        story_output += "-"
    story_output += '\n'

    # PERP ORG:
    story_output += "PERP ORG: "
    empty = True
    for word in story:
        word_vec = ie_model.word_features(word, features)
        result = classifier.run_perceptron(word_vec, labels["perporg"][0], labels["perporg"][1])
        if result > 0:
            story_output += word + '\n'
            empty = False
    if empty:
        story_output += "-"
    story_output += '\n'

    # TARGET:
    story_output += "TARGET: "
    empty = True
    for word in story:
        word_vec = ie_model.word_features(word, features)
        result = classifier.run_perceptron(word_vec, labels["target"][0], labels["target"][1])
        if result > 0:
            story_output += word + '\n'
            empty = False
    if empty:
        story_output += "-"
    story_output += '\n'

    # VICTIM:
    story_output += "VICTIM: "
    empty = True
    for word in story:
        word_vec = ie_model.word_features(word, features)
        result = classifier.run_perceptron(word_vec, labels["victim"][0], labels["victim"][1])
        if result > 0:
            story_output += word + '\n'
            empty = False
    if empty:
        story_output += "-"
    story_output += '\n\n'

    return story_output


def dic_to_string(weights, bias):
    string = str(bias)
    for key, value in weights.items():
        string = string + " " + str(key) + ':' + str(value)
    return string


def string_to_dic(string):
    string = string.split()
    dic = {}
    for x in range(1, len(string)):
        entry = string[x].split(':')
        dic[int(entry[0])] = float(entry[1])
    return dic, float(string[0])


def run():

    # Convert texts to vector data with labels based on answers
    # This also returns a dict of (feature, index) where there are features for
    # all words in the training corpus, all pos tags in the training corpus, and
    # the incidents list

    # Run this to create training vectors and store feature data again
    # ie_model.run()

    # Recover features from file
    features = {}
    with open("feature_indices.txt") as file:
        feat = file.read().split()
        for f in feat:
            f = f.split(':')
            features[f[0]] = int(f[1])

    # Labels for our classifiers
    labels = {"arson": '', "attack": '', "bombing": '', "kidnapping": '', "robbery": '', "weapon": '',
              "perpindiv": '', "perporg": '', "target": '', "victim": ''}

    # Calculate and store weights and biases for all labels
    # for label in labels:
    #     print("Calculating Weights and Biases for Label: " + label)
    #     weights, bias = classifier.run(label + ".txt.vector", features)
    #     weight_string = dic_to_string(weights, bias)
    #     with open("weights/" + label + ".txt.trained", 'w') as file:
    #         file.write(weight_string)

    # Recover weights and biases from file
    for label in labels:
        with open("weights/" + label + ".txt.trained") as file:
            weights, bias = string_to_dic(file.read())
            labels[label] = (weights, bias)

    # Load test file
    test_file = sys.argv[1]

    with open(test_file) as file:
        data = file.read()

    data = data.splitlines()
    story = ''
    results = ''
    for line in data:
        if not line:
            continue
        if len(line.split()[0]) > 2:
            if line.split()[0][:3] == "DEV" or line.split()[0][:3] == "TST":
                if story != '':
                    results += label_story(story, labels, features)
                    story = ''
                story += line + '\n'
            else:
                story += line + '\n'

    results += label_story(story, labels, features)

    with open(test_file + '.templates', 'w') as file:
        file.write(results)


run()
