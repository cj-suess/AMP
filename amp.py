import numpy as np
import matplotlib.pyplot as plt

def generate_sk_matrix(N):
    # mean 0, variance 1/N scaled for symmetry
    J = np.random.normal(loc=0.0, scale=1.0/np.sqrt(N), size=(N, N))
    J = (J + J.T) / np.sqrt(2)
    np.fill_diagonal(J, 0.0)
    return J

def calculate_energy(sigma, J):
    # the Hamiltonian: -0.5 * sigma^T * J * sigma
    return -0.5 * (sigma.T @ J @ sigma)

def gradient_descent_sk(J, num_iterations=200, learning_rate=0.1):
    N = J.shape[0]
    # start with random continuous spins between -1 and 1
    # this is our random drop point in the mountain range
    sigma = np.random.uniform(-1.0, 1.0, size=N)
    
    # track elevation over time
    energy_history = []
    
    for i in range(num_iterations):
        # calculate the gradient
        gradient = -J @ sigma
        # take a step downhill
        sigma = sigma - (learning_rate * gradient)
        # project back to reality (clip values to strictly stay between -1 and 1)
        sigma = np.clip(sigma, -1.0, 1.0)
        # calculate current energy
        current_energy = -0.5 * (sigma.T @ J @ sigma)
        energy_history.append(current_energy)
        
    return sigma, energy_history

# optimized damped AMP
# using field-damping to prevent the numerical instabilities
def amp_sk(J, m_init, num_iterations=1000, damping=0.7):
  
    N = J.shape[0]
    # small initial belief to break symmetry (Passed in via m_init)
    m = np.copy(m_init)
    m_old = np.zeros(N)
    h = np.zeros(N)
    
    # smooth cooling schedule up to the algorithmic threshold
    betas = np.linspace(0.1, 2.5, num_iterations) 
    
    for beta in betas:
        # onsager correction term is key for high-dimensional random matrices
        onsager_coef = np.mean(1.0 - m**2)
        h_target = J @ m - beta * onsager_coef * m_old
        
        # field damping acts as a numerical shock absorber
        h = damping * h + (1.0 - damping) * h_target
        m_old = np.copy(m)
        m = np.tanh(beta * h)
            
    return np.sign(m)

# local search flips individual spins to find the exact local minimum of the current valley.
def greedy_quench(sigma, J):
    sigma_opt = np.copy(sigma).astype(float)
    N = len(sigma_opt)
    improved = True
    
    while improved:
        improved = False
        # push from all other spins
        local_fields = J @ sigma_opt
        # if sigma_i * field_i < 0, the spin is 'frustrated'
        frustration = sigma_opt * local_fields
        
        # flip the most frustrated spin
        idx = np.argmin(frustration)
        if frustration[idx] < 0:
            sigma_opt[idx] *= -1
            improved = True
            
    return sigma_opt

# generates a basis of perpendicular starting vectors
def get_orthogonal_starts(num_starts, N, scale=0.001):
    # create random directions, then use QR decomposition to make them perpendicular
    random_directions = np.random.normal(size=(N, num_starts))
    Q, _ = np.linalg.qr(random_directions)
    # return the columns of Q (transposed to rows) scaled to the starting belief size
    return Q.T[:num_starts] * scale

N = 1000
num_restarts = 100 # explore n different random start points
best_energy = np.inf
best_spins = None

print(f"Generating SK Model (N={N})...")
J_matrix = generate_sk_matrix(N)
parisi_val = -0.7633 * N # the theoretical floor

# run GD
gd_final_spins, gd_energy_history = gradient_descent_sk(J_matrix)

# generate the set of orthogonal starting points before the loop begins
print(f"Generating {num_restarts} Orthogonal Starting Points...")
orthogonal_starts = get_orthogonal_starts(num_restarts, N)

print(f"Starting Multi-Start Sweep ({num_restarts} runs)...")

for i in range(num_restarts):
    # run message passing
    raw_spins = amp_sk(J_matrix, m_init=orthogonal_starts[i])
    
    # perform the greedy "Polish"
    quenched_spins = greedy_quench(raw_spins, J_matrix)
    
    current_energy = calculate_energy(quenched_spins, J_matrix)
    
    if current_energy < best_energy:
        best_energy = current_energy
        best_spins = quenched_spins
    
    print(f"  Run {i+1:03d}: Energy {current_energy:8.2f} | Gap to Limit: {abs(current_energy - parisi_val):8.2f}")

print("\n" + "="*45)
print(f"FINAL BEST ENERGY: {best_energy:.4f}")
print(f"THEORETICAL LIMIT: {parisi_val:.4f}")
print(f"EFFICIENCY:        {100 * (best_energy/parisi_val):.2f}% of optimal")
print("="*45)

print(f"Gradient Descent Final Energy: {gd_energy_history[-1]:.4f}")