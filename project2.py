import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Original sample data
# ------------------------------
X = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69])
n = len(X)
mu_hat = np.mean(X)
print("Sample mean (mu_hat):", mu_hat)

# ------------------------------
# 2. Bootstrap function
# ------------------------------
def bootstrap_probability(x, nboot, mu_hat, a=-6, b=4):
    record_means = []   # to store bootstrap sample means
    count = 0           # to count how many differences fall within (a, b)
    
    for j in range(nboot):
        # Generate bootstrap sample with replacement
        index = np.random.randint(0, len(x), len(x))
        sample = x[index]
        mean_sample = np.mean(sample)
        record_means.append(mean_sample)

        # Check if the difference lies between a and b
        diff = mean_sample - mu_hat
        if a < diff < b:
            count += 1

        # Print first few iterations to see what’s happening
        if j < 5:
            print(f"Iteration {j+1}: index={index}, sample={sample}, mean={mean_sample:.2f}, diff={diff:.2f}")
    
    # Estimate probability
    estimated_p = count / nboot
    print(f"\nEstimated probability p ≈ {estimated_p:.4f}")
    
    return np.array(record_means), estimated_p

# ------------------------------
# 3. Run the bootstrap
# ------------------------------
nboot = 600
resultBoots, p_hat = bootstrap_probability(X, nboot, mu_hat)

# ------------------------------
# 4. Plot histogram
# ------------------------------
plt.hist(resultBoots, bins=30, color='skyblue', edgecolor='black')
plt.title("Bootstrap distribution of sample means")
plt.xlabel("Sample mean")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
