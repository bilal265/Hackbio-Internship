# 1) Function for translating DNA to protein
# Define the genetic code as a constant 

GENETIC_CODE = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
    'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
}
# Input validation
    seq = seq.upper()
    if not all(base in 'ATCG' for base in seq):
        raise ValueError("Sequence contains invalid nucleotides. Use only A, T, C, G.")
    if len(seq) % 3 != 0:
        raise ValueError("Sequence length must be a multiple of 3 for a complete reading frame.")

    translated_protein = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        amino_acid = GENETIC_CODE[codon]
        translated_protein.append(amino_acid)
        if amino_acid == '*':  # Stop translation at stop codon
            break

    return ''.join(translated_protein)

    # Test sequence (e.g., ATG codon for Methionine)
    dna_seq = "ATGTAA"
    try:
        protein = dna_to_protein(dna_seq)
        print(f"DNA: {dna_seq}")
        print(f"Protein: {protein}")
    except ValueError as e:
        print(f"Error: {e}")

# 2) Function for stimulation and generation of bacterial population growth curve

import numpy as np
import pandas as pd

def simulate_logistic_growth(
    r_max=0.1,  # Maximum growth rate
    K=1000,     # Carrying capacity
    N0=10,      # Initial population
    total_time=100,  # Total simulation time
    lag_phase_range=(5, 15),  # Range for random lag phase duration
    exp_phase_range=(20, 40),  # Range for random exponential phase duration
    time_step=1  # Time step for simulation
):
  time_points = np.arange(0, total_time, time_step)
    
    # Randomize lag phase and exponential phase durations
    lag_duration = np.random.uniform(*lag_phase_range)
    exp_duration = np.random.uniform(*exp_phase_range)
    
    # Initialize population array
    N = np.zeros(len(time_points))
    N[0] = N0
    
    # Simulate growth
    for t in range(1, len(time_points)):
        current_time = time_points[t]
        
        # Lag phase: no growth until lag_duration
        if current_time < lag_duration:
            N[t] = N[t-1]
        else:
            # Exponential phase: growth accelerates until exp_duration after lag
            effective_time = current_time - lag_duration
            if effective_time < exp_duration:
                # Use a simplified logistic growth approximation
                dN = r_max * N[t-1] * (1 - N[t-1] / K)
                N[t] = N[t-1] + dN * time_step
            else:
                # Logistic growth after exponential phase
                dN = r_max * N[t-1] * (1 - N[t-1] / K)
                N[t] = N[t-1] + dN * time_step
    
    # Ensure population doesn't exceed carrying capacity
    N = np.minimum(N, K)
    
    return time_points, N

# Generate 100 different growth curves
n_curves = 100
data = {}

for i in range(n_curves):
    time_points, population = simulate_logistic_growth()
    data[f'Curve_{i}'] = population

# Convert to DataFrame
df = pd.DataFrame(data, index=time_points)
df.index.name = 'Time'

# Display the first few rows of the DataFrame
print(df.head())

#Save to CSV for further analysis
df.to_csv('logistic_growth_curves.csv')

# 3) Function for determining the time to reach 80% of the maximum growth

def time_to_80_percent_max(simulate_func, r_max=0.1, K=1000, N0=10, total_time=100, 
                          lag_phase_range=(5, 15), exp_phase_range=(20, 40), time_step=1):
                            
        # Simulate the growth curve
    time_points, population = simulate_func(
        r_max=r_max,
        K=K,
        N0=N0,
        total_time=total_time,
        lag_phase_range=lag_phase_range,
        exp_phase_range=exp_phase_range,
        time_step=time_step
    )
    
    # Target population (80% of carrying capacity)
    target_N = 0.8 * K
    
    # Find the index where population crosses 80% of K
    idx = np.where(population >= target_N)[0]
    if len(idx) == 0:
        return -1  # 80% of K not reached within simulation time
    
    idx = idx[0]  # First index where population >= target_N
    
    # If already at or above target at the start, return the first time point
    if idx == 0:
        return time_points[0]
    
    # Linear interpolation between the previous and current points
    prev_idx = idx - 1
    prev_time = time_points[prev_idx]
    prev_pop = population[prev_idx]
    curr_time = time_points[idx]
    curr_pop = population[idx]
    
    # Interpolate time
    if curr_pop == prev_pop:  # Avoid division by zero
        return curr_time
    slope = (curr_time - prev_time) / (curr_pop - prev_pop)
    interpolated_time = prev_time + slope * (target_N - prev_pop)
    
    return interpolated_time
                            
# Calculate time to reach 80% of K
    time_80 = time_to_80_percent_max(simulate_logistic_growth)
    print(f"Time to reach 80% of carrying capacity (K={1000}): {time_80:.2f} units")

    # Generate and test with multiple curves
    times_to_80 = []
    for _ in range(10):  # Test with 10 random curves
        time_80 = time_to_80_percent_max(simulate_logistic_growth)
        times_to_80.append(time_80)
    print(f"Times to 80% of K for 10 curves: {times_to_80}")  

# 4)write a function for calculating the hamming distance between your Slack username and twitter/X handle (synthesize if you don’t have one). Feel free to pad it with extra words if they are not of the same length.

def hamming_distance(str1, str2, pad_char='_'):
  # Input validation
    if not isinstance(str1, str) or not isinstance(str2, str):
        raise ValueError("Both inputs must be strings.")
    
    # Determine the longer length and pad the shorter string
    len1, len2 = len(str1), len(str2)
    max_length = max(len1, len2)
    
    # Pad the shorter string with pad_char
    str1_padded = str1.ljust(max_length, pad_char)
    str2_padded = str2.ljust(max_length, pad_char)
    
    # Calculate Hamming distance
    distance = sum(c1 != c2 for c1, c2 in zip(str1_padded, str2_padded))
    
    return distance

# Example usage with your Slack username and Instagram handle
slack_username = "bilal ashraf"
insta_handle = "mohammadbilalashraf15"

distance = hamming_distance(slack_username, insta_handle)
print(f"Hamming distance between '{slack_username}' and '{insta_handle}': {distance}")

# Optional: Print padded strings for clarity
max_len = max(len(slack_username), len(insta_handle))
slack_padded = slack_username.ljust(max_len, '_')
insta_padded = insta_handle.ljust(max_len, '_')
print(f"Padded strings: '{slack_padded}' and '{insta_padded}'")
