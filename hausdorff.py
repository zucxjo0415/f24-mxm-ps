import numpy as np
from scipy.linalg import eig
from scipy.optimize import root_scalar
from cmath import phase
from geometry_tools import hyperbolic

# Define Möbius transformations
class MobiusTransform:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float64)

    def apply(self, pt):
        """Takes a hyperbolic.Point and returns the transformed point"""
        a, b, c, d = self.matrix.flatten()
        plane_coords = pt.coords(model='poincare')
        z = plane_coords[0]+1j*plane_coords[1]
        transformed =  (a * z + b) / (c * z + d)
        disk_coords = hyperbolic.Point([transformed.real, transformed.imag], model='poincare')
        return disk_coords
    
    def apply_inverse(self, pt):
        a, b, c, d = np.linalg.inv(self.matrix).flatten()
        plane_coords = pt.coords(model='poincare')
        z = plane_coords[0]+1j*plane_coords[1]
        transformed =  (a * z + b) / (c * z + d)
        disk_coords = hyperbolic.Point([transformed.real, transformed.imag], model='poincare')
        return disk_coords

    def derivative(self, pt):
        a, b, c, d = self.matrix.flatten()
        plane_coords = pt.coords(model='poincare')
        z = plane_coords[0]+1j*plane_coords[1]
        denom = (c * z + d)**2
        return abs(a * d - b * c) / abs(denom)
    
class Point:
    def __init__(self, pt, radius, f):
        self.pt = pt # geometry_tools point
        self.radius = radius # Neighborhood radius
        self.f = f # Which generator to use

# Define generators
A = np.array([[3, 0], [0, 1/3]], dtype=np.float64)
B = np.array([[5/3, 4/3], [4/3, 5/3]], dtype=np.float64)
A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)
generators = [MobiusTransform(B_inv), MobiusTransform(A_inv), MobiusTransform(B), MobiusTransform(A)]

sample_points = [hyperbolic.Point([x,y], model='poincare') for (x,y) in [(1,0), (0,1), (-1,0), (0,-1)]]
points = [Point(sample_points[i], radius=np.pi/4, f=generators[i]) for i in range(len(sample_points))]


# TODO: matrix is way sparser than it should be
def compute_transition_matrix(points):
    """Creates the transition matrix for the given generators and sample points"""
    n = len(points)
    T = np.zeros((n, n), dtype=np.float64)
    refinement = []
    for i, x_i in enumerate(points):
        for j, x_j in enumerate(points):
            try:
                f = x_i.f
                x_mapped = f.apply(x_i.pt)
                #This is checking i -> j from the paper
                x_mapped_coords, x_j_coords = x_mapped.coords(model='poincare'), x_j.pt.coords(model='poincare')
                x_mapped_phase, x_j_phase = phase(x_mapped_coords[0]+1j*x_mapped_coords[1]), phase(x_j_coords[0]+1j*x_j_coords[1])
                if np.abs(x_mapped_phase - x_j_phase) < x_j.radius:
                    y_ij = f.apply_inverse(x_j.pt)
                    derivative = f.derivative(y_ij)
                    T[i, j] = 1 / np.abs(derivative)
                    # Might as well compute refinement here
                    refinement.append(Point(f.apply_inverse(x_j.pt), x_j.radius / 3, f))
            except Exception as e:
                print(e)
                continue
    return T, refinement

# Find alpha(P) such that the spectral radius of T^alpha is 1
def find_alpha(T):
    def spectral_radius(alpha):
        T_alpha = np.power(T, alpha)
        w, _ = eig(T_alpha)
        return max(abs(w)) - 1
    print(T)
    print(spectral_radius(0))
    print(spectral_radius(1))
    result = root_scalar(spectral_radius, bracket=[0, 1.1], method='brentq')
    print(f"Alpha: {result.root:.6f}")
    return result.root


def hausdorff_dimension(points, max_iter=10, tol=1e-6):
    for _ in range(max_iter):
        T, points = compute_transition_matrix(points)
        alpha = find_alpha(T)
    return alpha

if __name__ == "__main__":
    dim = hausdorff_dimension(points)
    print(f"Hausdorff Dimension: {dim:.6f}")
