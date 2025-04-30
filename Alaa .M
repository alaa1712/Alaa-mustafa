X = [
    [1, 0, 0, 0],  
    [0, 1, 0, 0],  
    [0, 0, 1, 0],  
]
Y = [0, 0, 0, 1]  

def sigmoid(x):
    return [1 / (1 + 2.71828 ** (-i)) for i in x]

def dsigmoid(y):
    return [i * (1 - i) for i in y]

def softmax(x):
    exps = [2.71828 ** i for i in x]
    total = sum(exps)
    return [i / total for i in exps]

def matmul(a, b):
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            val = 0
            for k in range(len(b)):
                val += a[i][k] * b[k][j]
            row.append(val)
        result.append(row)
    return result
input_size = 4
hidden_size = 4
output_size = 4
lr = 0.1

Wxh = [[0.4 for _ in range(hidden_size)] for _ in range(input_size)]
Whh = [[0.1 for _ in range(hidden_size)] for _ in range(hidden_size)]
Why = [[0.3 for _ in range(output_size)] for _ in range(hidden_size)]
bh = [0 for _ in range(hidden_size)]
by = [0 for _ in range(output_size)]

hs = [[0 for _ in range(hidden_size)]]

for t in range(len(X)):
    xt = [X[t]]
    h_prev = [hs[-1]]

    xh = matmul(xt, Wxh)
    hh = matmul(h_prev, Whh)
    h_raw = [xh[0][i] + hh[0][i] + bh[i] for i in range(hidden_size)]
    h = sigmoid(h_raw)
    hs.append(h)

y_raw = [sum(hs[-1][i] * Why[i][j] for i in range(hidden_size)) + by[j] for j in range(output_size)]
y_pred = softmax(y_raw)

dy = [y_pred[i] - Y[i] for i in range(output_size)]

for i in range(hidden_size):
    for j in range(output_size):
        Why[i][j] -= lr * hs[-1][i] * dy[j]
for j in range(output_size):
    by[j] -= lr * dy[j]

dh = [sum(dy[j] * Why[i][j] for j in range(output_size)) for i in range(hidden_size)]
dh_raw = [dh[i] * dsigmoid([hs[-1][i]])[0] for i in range(hidden_size)]

for i in range(input_size):
    for j in range(hidden_size):
        Wxh[i][j] -= lr * X[-1][i] * dh_raw[j]
for i in range(hidden_size):
    for j in range(hidden_size):
        Whh[i][j] -= lr * hs[-2][i] * dh_raw[j]
for j in range(hidden_size):
    bh[j] -= lr * dh_raw[j]

print("\nPredicted Output (softmax):")
for i, prob in enumerate(y_pred):
    print(f"Class {i}: {round(prob, 4)}")
