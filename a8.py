from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]



print("\n<<<<<<< 2 HIDDEN NODES >>>>>>>\n")

# After 9300 iterations, the change in error converged by reaching a value less than 0.0005
# The actual value after those iterations was 0.0004976351530286594
# I ran it with 50000 iterations and reached a change in error of 8.242983108703615e-05

xor_neural_net = NeuralNet(2, 2, 1)
xor_neural_net.train(xor_training_data, iters=50000)



print("\n<<<<<<< 8 HIDDEN NODES >>>>>>>\n")

# After 7700 iterations, the change in error converged by reaching a value less than 0.0005
# The actual value after those iterations was 0.0004958616290734465
# I ran it with 50000 iterations and reached a change in error of 6.354714023156375e-05

xor_neural_net = NeuralNet(2, 8, 1)
xor_neural_net.train(xor_training_data, iters=50000)



print("\n<<<<<<< 1 HIDDEN NODE >>>>>>>\n")

# After 50000 iterations, the change in error never converged and stayed at around 0.34
# This is because the XOR function cannot be computed with only one hidden layer

xor_neural_net = NeuralNet(2, 1, 1)
xor_neural_net.train(xor_training_data, iters=50000)



print("<<<<<<<<<<<<<< VOTER DATA >>>>>>>>>>>>>>\n")

voter_training_data = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]),
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]),
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0]),
]

voter_testing_inputs = [
    [1.0, 1.0, 1.0, 0.1, 0.1],
    [0.5, 0.2, 0.1, 0.7, 0.7],
    [0.8, 0.3, 0.3, 0.3, 0.8],
    [0.8, 0.3, 0.3, 0.8, 0.3],
    [0.9, 0.8, 0.8, 0.3, 0.6],
]

voter_neural_net = NeuralNet(5, 8, 1)
voter_neural_net.train(voter_training_data, iters=50000)

for inputs in voter_testing_inputs:
    classification = "Republican" if voter_neural_net.evaluate(inputs)[0] > 0.5 else "Democrat"
    print(f"Voter Inputs: {inputs}")
    print(f"Predicted Party: {classification}\n")