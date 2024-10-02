import numpy as np
import matplotlib.pyplot as plt

A = np.array([[3, 0], [0, 1/3]], dtype=np.float64)
A_inv = np.linalg.inv(A)
B = np.array([[5/3, 4/3], [4/3, 5/3]], dtype=np.float64)
B_inv = np.linalg.inv(B)

# Hyperbolic distance between two complex numbers
def hyperbolic_distance(z, w):
    numerator = np.abs(z - w)
    denominator = 2 * z.imag * w.imag
    return np.arccosh(1 + numerator / denominator)

# Generate a random element of the free group
def random_element(max_len):
    integers = np.arange(max_len)
    length = np.random.choice(integers, p=integers / np.sum(integers))
    res = np.identity(2)
    generators = [A, A_inv, B, B_inv]
    last_gen = -1
    for i in range(length):
        next_gen = np.random.randint(0, 4)
        while is_inverse(next_gen, last_gen):
            next_gen = np.random.randint(0, 4)
        res = generators[next_gen] @ res
        last_gen = next_gen
    return res

# This is so ugly pls delete soon
def is_inverse(i, j):
    return i == 1 and j == 0 or i == 0 and j == 1 or i == 3 and j == 2 or i == 2 and j == 3

# Enumerate all elements of the free group up to a certain length
def enumerate_free_group_elements(max_len):
    generators = [A, A_inv, B, B_inv]
    elements = []
    def helper(depth, word, last_added):
        if depth > max_len:
            return
        elements.append(word)
        for i in range(len(generators)):
            if is_inverse(i, last_added):
                continue
            helper(depth + 1, word @ generators[i], i)
    helper(0, np.identity(2), -1)
    return elements

# Randomly sample elements of the free group (useful for large N)
def enumerate_montecarlo(max_len, num_samples):
    elements = []
    for i in range(num_samples):
        elements.append(random_element(max_len))
    return elements

# Our group action on complex numbers from the notes (az + b) / (cz + d)
def mobius_transform(matrix, z):
    return (matrix[0, 0] * z + matrix[0, 1]) / (matrix[1, 0] * z + matrix[1, 1])

# The expression in the limit that we are trying to calculate
def limit_calculation(data, n):
    count = 0
    for dist in data:
        if dist < n:
            count += 1
    return np.log(count) / n

if __name__ == "__main__":
    o = 0 + 1j # basepoint
    N = 12 # maximum length of words


    if N <= 13: # Takes forever past length 13
        els = enumerate_free_group_elements(N)
    else:
        els = enumerate_montecarlo(N, 1000)
    data = []
    for el in els:
        dist = hyperbolic_distance(o, mobius_transform(el, o))
        data.append(dist)
    data = np.array(data)
    data = data[np.isfinite(data)]

    for n in range(5, 100, 5):
        print(f"n = {n}: {limit_calculation(data, n)}")


    plt.hist(data, bins=100)
    plt.title(f"Displacement of words up to length {N} (Basepoint = i)")
    plt.xlabel("Displacement (Hyperbolic Half Plane)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.savefig(f"figures/distribution{N}.png")
    plt.show()
