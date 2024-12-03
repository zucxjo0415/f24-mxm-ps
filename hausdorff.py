import numpy as np
from scipy.linalg import eig
from scipy.optimize import root_scalar

# Define Möbius transformations
class MobiusTransform:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float64)

    def apply(self, z):
        a, b, c, d = self.matrix.flatten()
        in_plane = disk_to_plane(z)
        transformed =  (a * in_plane + b) / (c * in_plane + d)
        return plane_to_disk(transformed)

    def derivative(self, z):
        a, b, c, d = self.matrix.flatten()
        denom = (c * z + d)**2
        return abs(a * d - b * c) / abs(denom)
    
class Point:
    def __init__(self, pos, radius):
        self.pos = pos # Complex number
        self.radius = radius # Neighborhood radius

    def __repr__(self):
        return f"Point({self.z}, {self.radius})"
    
def disk_to_plane(z):
    return (z-1j) / (z+1j)

def plane_to_disk(w):
    return 1j * (w + 1) / (w - 1)

# Define generators
A = np.array([[3, 0], [0, 1/3]], dtype=np.float64)
B = np.array([[5/3, 4/3], [4/3, 5/3]], dtype=np.float64)
A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)
generators = [MobiusTransform(A), MobiusTransform(B), MobiusTransform(A_inv), MobiusTransform(B_inv)]

# Trying out the angle method
angles = np.linspace(0, 2 * np.pi, 5)[:-1]
sample_points = [np.exp(1j * angle) for angle in angles]
points = [Point(z, radius=np.pi/4) for z in sample_points]


# TODO: matrix is way sparser than it should be
def compute_transition_matrix(generators, points):
    """Creates the transition matrix for the given generators and sample points"""
    n = len(points)
    T = np.zeros((n, n), dtype=np.float64)
    for i, z_i in enumerate(points):
        for j, z_j in enumerate(points):
            for f in generators:
                try:
                    # Check where z_i maps to under f
                    z_mapped = f.apply(z_i.pos)
                    if np.abs(z_mapped-z_j.pos) < z_j.radius:
                        derivative = f.derivative(z_i.pos)
                        T[i, j] = 1 / np.abs(derivative)
                except ZeroDivisionError:
                    print("Zero division error")
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
def refine_partition(points):
    new_points = []
    for i in range(len(points)):
        z1 = points[i]
        z2 = points[(i + 1) % len(points)]
        midpoint = (z1.pos + z2) / abs(z1.pos + z2.pos)  # Project to the unit circle
        new_points.append(Point(z1.pos, z1.radius / 2))
        new_points.append(Point(midpoint, z1.radius / 2))
    return np.array(new_points)

def hausdorff_dimension(generators, points, max_iter=10, tol=1e-6):
    for _ in range(max_iter):
        T = compute_transition_matrix(generators, points)
        alpha = find_alpha(T)
        points = refine_partition(points)
    return alpha

if __name__ == "__main__":
    dim = hausdorff_dimension(generators, points)
    print(f"Hausdorff Dimension: {dim:.6f}")
