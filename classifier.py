import random


def load_file(file_name):
    with open(file_name) as f:
        data = f.read()
    return data


def num_features(data):
    max_features = 0
    data = data.splitlines()
    for example in data:
        example = example.split()
        max_example = int(example[len(example)-1].split(':')[0])
        if max_example > max_features:
            max_features = max_example
    return max_features


def process_data(data):
    data = data.splitlines()

    examples = []

    for line in data:
        line = line.split()
        example = {}
        if len(line) == 0:
            print("Length of example: " + str(len(line)))
        if line[0] != '-1' and line[0] != '+1':
            print("No Label: " + str(line))
        for x in range(len(line)):
            if x == 0:
                if line[0] == "-1":
                    example[x] = -1
                else:
                    example[x] = 1
            else:
                entry = line[x].split(':')
                target = int(entry[0])
                value = float(entry[1])
                example[target] = value
        examples.append(example)
    return examples


def dot_prod(vec1, vec2):
    sum = 0
    for key, value in vec1.items():
        if key in vec2:
            sum += (vec2[key] * value)
    return sum


def simple_perceptron(examples, features, rate, epochs, total=True, updates=False):

    w = {}

    epoch_weights = []
    epoch_biases = []

    for x in range(features):
        w[x] = random.uniform(-.01, .01)

    b = random.uniform(-.01, .01)

    num_updates = 0

    for x in range(epochs):
        random.shuffle(examples)
        for example in examples:
            label = example[0]
            example.pop(0, None)

            # predict the examples outcome with the weight vector
            outcome = dot_prod(example, w) + b

            if outcome * label <= 0:
                #update = [x * rate * label for key, x in example]
                for key, val in example:
                    w[key] += val * rate * label
                b = b + (rate * label)
                num_updates += 1
        epoch_weights.append(w)
        epoch_biases.append(b)

    if updates:
        print(num_updates/epochs)

    if total:
        return w, b
    else:
        return epoch_weights, epoch_biases


def dynamic_perceptron(examples, features, rate, epochs, total=True, updates=False):

    w = {}

    epoch_weights = []
    epoch_biases = []

    for x in range(1, len(features)+1):
        w[x] = random.uniform(-.01, .01)

    b = random.uniform(-.01, .01)

    count = 0
    num_updates = 0

    for x in range(epochs):
        random.shuffle(examples)
        for example in examples:

            dyn_rate = rate / (1 + count)
            count += 1

            label = example[0]
            example.pop(0, None)

            # predict the examples outcome with the weight vector
            outcome = dot_prod(example, w) + b

            if outcome * label <= 0:
                #update = [x * dyn_rate * label for x in features]
                for key, value in example.items():
                    w[key] += x * dyn_rate * label
                #w = [sum(x) for x in zip(w, update)]
                b = b + (dyn_rate * label)
                num_updates += 1
            example[0] = label
        epoch_weights.append(w)
        epoch_biases.append(b)

    if updates:
        print(num_updates/epochs)

    if total:
        return w, b
    else:
        return epoch_weights, epoch_biases


def margin_perceptron(examples, features, rate, epochs, margin, total=True, updates=False):

    w = []

    epoch_weights = []
    epoch_biases = []

    for x in range(features):
        w.append(random.uniform(-.01, .01))

    b = random.uniform(-.01, .01)

    count = 0
    num_updates = 0

    for x in range(epochs):
        random.shuffle(examples)
        for example in examples:

            dyn_rate = rate / (1 + count)
            count += 1

            features = example[1:]
            label = example[0]

            # predict the examples outcome with the weight vector
            outcome = dot_prod(features, w) + b

            if outcome * label <= margin:
                update = [x * dyn_rate * label for x in features]
                w = [sum(x) for x in zip(w, update)]
                b = b + (dyn_rate * label)
                num_updates += 1
        epoch_weights.append(w)
        epoch_biases.append(b)

    if updates:
        print(num_updates/epochs)

    if total:
        return w, b
    else:
        return epoch_weights, epoch_biases


def averaged_perceptron(examples, features, rate, epochs, total=True, updates=False):

    w = []

    epoch_weights = []
    epoch_biases = []

    for x in range(features):
        w.append(random.uniform(-.01, .01))

    b = random.uniform(-.01, .01)

    w_a = [0] * features
    b_a = 0

    num_updates = 0

    for x in range(epochs):
        random.shuffle(examples)
        for example in examples:
            features = example[1:]
            label = example[0]

            # predict the examples outcome with the weight vector
            outcome = dot_prod(features, w) + b

            if outcome * label <= 0:
                update = [x * rate * label for x in features]
                w = [sum(x) for x in zip(w, update)]
                b = b + (rate * label)
                num_updates += 1

            w_a = [sum(x) for x in zip(w, w_a)]
            b_a = b_a + b
        examples_num = len(examples) * (x+1)
        epoch_weights.append([x/examples_num for x in w_a])
        epoch_biases.append(b_a/examples_num)

    if updates:
        print(num_updates/epochs)

    if total:
        return [x/(len(examples)*epochs) for x in w_a], b_a/(len(examples)*epochs)
    else:
        return epoch_weights, epoch_biases


def test_perceptron(tests, weights, bias):

    num_tests = 0
    successes = 0

    for test in tests:
        num_tests += 1
        label = test[0]
        features = test[1:]

        result = dot_prod(features, weights) + bias

        if result * label >= 0:
            successes += 1

    return successes / num_tests


def run_perceptron(features, weights, bias):

    result = dot_prod(features, weights) + bias

    if result > 0:
        return True
    else:
        return False


def cross_validate_perceptron(mode, rate, epochs, margin=1):
    k0 = load_file("Dataset/CVSplits/training00.data")
    k1 = load_file("Dataset/CVSplits/training01.data")
    k2 = load_file("Dataset/CVSplits/training02.data")
    k3 = load_file("Dataset/CVSplits/training03.data")
    k4 = load_file("Dataset/CVSplits/training04.data")

    data = [k0, k1, k2, k3, k4]

    features = 0
    for set in data:
        num = num_features(set)
        if num > features:
            features = num

    d0 = process_data(k0, features)
    d1 = process_data(k1, features)
    d2 = process_data(k2, features)
    d3 = process_data(k3, features)
    d4 = process_data(k4, features)

    data = [d0, d1, d2, d3, d4]

    succ_rates = []

    for i in range(0, 5):
        test_data = data[i]
        training_data = []
        for j in range(0, 5):
            if j is not i:
                training_data += data[j]
        if mode == "simple":
            weights, bias = simple_perceptron(training_data, features, rate, epochs)
        elif mode == "dynamic":
            weights, bias = dynamic_perceptron(training_data, features, rate, epochs)
        elif mode == "margin":
            weights, bias = margin_perceptron(training_data, features, rate, epochs, margin)
        elif mode == "average":
            weights, bias = averaged_perceptron(training_data, features, rate, epochs)

        succ_rates.append(test_perceptron(test_data, weights, bias))

    return sum(succ_rates) / len(succ_rates)


def run(training_data, features):

    data = load_file("weights/" + training_data)

    data = process_data(data)

    # modes = ["simple", "dynamic", "margin", "average"]
    # learning_rates = [1, 0.1, 0.01]
    # margins = [1, 0.1, 0.01]

    best_rate = {"simple": 0.01, "dynamic": 0.1, "margin": 1, "margin_margin": 0.1, "average": 0.1}

    # Train each mode over 20 epochs using training data and report accuracy against dev data
    #for mode in modes:
    mode = "dynamic"
    if mode == "simple":
        weights, bias = simple_perceptron(data, features, best_rate[mode], 20, total=False)
    if mode == "dynamic":
        weights, bias = dynamic_perceptron(data, features, best_rate[mode], 20)
    if mode == "margin":
        weights, bias = margin_perceptron(data, features, best_rate[mode], 20, best_rate["margin_margin"], total=False)
    if mode == "average":
        weights, bias = averaged_perceptron(data, features, best_rate[mode], 20, total=False)
    return weights, bias
