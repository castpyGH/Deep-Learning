import numpy as np

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

STEP_FUNCTION = stepFunction(-1)


def sigmoidFunction (soma):
    return 1/ (1 + np.exp(-soma))

SIGMOID_FUNCTION = sigmoidFunction(2.1)


def tahnFunction (soma):
    return (np.exp(soma) - np.exp(-soma) / np.exp(soma) + np.exp(-soma))

TAHN_FUNCTION = tahnFunction(2.1)


def reluFunction (soma):
    if (soma >= 0):
        return soma
    return 0

RELU_FUNCTION = reluFunction(2.1)


def linearFunction(soma):
    return soma

LINEAR_FUNCTION = linearFunction(2.1)


def softmaxFunction (x):
    ex = np.exp(x)
    return ex / ex.sum()

valoresDeEntrada= [5.0, 2.0, 1.3]
print(softmaxFunction(valoresDeEntrada))