import numpy as np
from scipy.linalg import eig
from scipy.optimize import root_scalar

# Define Möbius transformations
class MobiusTransform:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float64)

    def apply(self, z):
        a, b, c, d = self.matrix.flatten()
        return (a * z + b) / (c * z + d)

    def derivative(self, z):
        a, b, c, d = self.matrix.flatten()
        denom = (c * z + d)**2
        return abs(a * d - b * c) / abs(denom)

# Define generators
A = np.array([[3, 0], [0, 1/3]], dtype=np.float64)
B = np.array([[5/3, 4/3], [4/3, 5/3]], dtype=np.float64)
A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)
generators = [MobiusTransform(A), MobiusTransform(B), MobiusTransform(A_inv), MobiusTransform(B_inv)]

# Trying out the angle method
angles = np.linspace(0, 2 * np.pi, 5)[:-1]
sample_points = [np.exp(1j * angle) for angle in angles]


# TODO: matrix is way sparser than it should be
def compute_transition_matrix(generators, sample_points):
    """Creates the transition matrix for the given generators and sample points"""
    n = len(sample_points)
    T = np.zeros((n, n), dtype=np.float64)
    for i, z_i in enumerate(sample_points):
        for j, z_j in enumerate(sample_points):
            for g in generators:
                try:
                    # Check where z_i maps to under g
                    z_mapped = g.apply(z_i)
                    if np.isclose(z_mapped, z_j):
                        derivative = g.derivative(z_i)
                        T[i, j] = 1 / np.abs(derivative)
                except ZeroDivisionError:
                    continue
    return T

# Find alpha(P) such that the spectral radius of T^alpha is 1
def find_alpha(T):
    def spectral_radius(alpha):
        T_alpha = T**alpha
        w, _ = eig(T_alpha)
        return max(abs(w)) - 1
    print(T)
    result = root_scalar(spectral_radius, bracket=[0, 1.5], method='brentq')
    print(f"Alpha: {result.root:.6f}")
    return result.root

# Refinement step
def refine_partition(sample_points):
    new_points = []
    for i in range(len(sample_points)):
        z1 = sample_points[i]
        z2 = sample_points[(i + 1) % len(sample_points)]
        midpoint = (z1 + z2) / abs(z1 + z2)  # Project to the unit circle
        new_points.append(z1)
        new_points.append(midpoint)
    return np.array(new_points)

def hausdorff_dimension(generators, sample_points, max_iter=10, tol=1e-6):
    for _ in range(max_iter):
        T = compute_transition_matrix(generators, sample_points)
        alpha = find_alpha(T)
        sample_points = refine_partition(sample_points)
    return alpha

if __name__ == "__main__":
    dim = hausdorff_dimension(generators, sample_points)
    print(f"Hausdorff Dimension: {dim:.6f}")
