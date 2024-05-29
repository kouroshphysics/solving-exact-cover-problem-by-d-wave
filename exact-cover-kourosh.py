import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
token="DEV-abdc4c7339474eacea7a99b9e4269e22f667d3f2"
solver = 'Advantage_system6.4'
# Example sets
sets = [{1, 2, 3}, {4}, {5,1}, { 5, 4}, {2,3}]
elements = set(e for s in sets for e in s)
n = len(sets)

# Hamiltonian coefficients
h = np.zeros(n)
J = np.zeros((n, n))

# Constants
A = 20
B = 1  #Penalty term for the total subsets included

#Create constraints and objective terms
for e in elements:
    indices = [i for i, s in enumerate(sets) if e in s]
    k_j = len(indices)
    for i in indices:
        h[i] += A * (k_j - 2) / 4
        for j in indices:
            if i != j:
                J[i, j] += A / 4

# Adding penalty term
for i in range(n):
    h[i] += B / 2

# Convert h and J to dictionaries for dimod
h_dict = {i: h[i] for i in range(n)}                     
J_dict = {(i, j): J[i, j] for i in range(n) for j in range(i + 1, n)}           

# Define the BQM
bqm = dimod.BinaryQuadraticModel.from_ising(h_dict, J_dict)

# Solve using D-Wave
sampler = EmbeddingComposite(DWaveSampler(token=token, solver=solver))
response = sampler.sample(bqm, num_reads=200)

# Extract and print solutions
print(response)
solution = response.first.sample

selected_sets = [sets[i] for i, val in solution.items() if val == 1]
print("Solution:", selected_sets)
  

# Solve exactly using ExactSolver
exact_solver = dimod.ExactSolver()
exact_response = exact_solver.sample(bqm)

# Extract and print solutions from ExactSolver
print("Solutions from ExactSolver:")
exact_solution=exact_response.first.sample
selected_sets_exact = [sets[i] for i, val in exact_solution.items() if val == 1]
print("exact Solution:", selected_sets_exact)
  


