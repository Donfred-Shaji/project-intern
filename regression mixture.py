import pandas as pd

# Load the dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Count occurrences of each category in the 'Like' column
like_counts = mcdonalds['Like'].value_counts()

# Reverse the order of the counts
reversed_like_counts = like_counts[::-1]

# Display the reversed counts
print(reversed_like_counts)
mcdonalds['Like_numeric'] = pd.to_numeric(mcdonalds['Like'], errors='coerce')

# Create a new column 'Like.n'
mcdonalds['Like.n'] = 6 - mcdonalds['Like_numeric']

# Count occurrences of each value in 'Like.n'
like_n_counts = mcdonalds['Like.n'].value_counts()

# Display the counts
print(like_n_counts)