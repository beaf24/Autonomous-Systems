import numpy as np

# Create a NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Save the array to a text file
np.savetxt('array.txt', arr)

print("Array saved to 'array.txt' file.")
