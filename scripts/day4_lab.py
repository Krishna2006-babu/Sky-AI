import numpy as np # NumPy is the math library for grids/images

# 1. Create a 5x5 "Image" (Grid of numbers)
# Imagine this is a tiny photo where 0=Black and 1=White
grid = np.array([
    [10, 20, 30, 40, 50],  # Row 0
    [11, 21, 31, 41, 51],  # Row 1
    [12, 22, 32, 42, 52],  # Row 2
    [13, 23, 33, 43, 53],  # Row 3
    [14, 24, 34, 44, 54]   # Row 4
])

print("--- ORIGINAL GRID ---")
print(grid)

# 2. THE CROP (Slicing)
# We want the center square (21, 31, 22, 32)
# Rows: 1 to 3 (remember, it stops BEFORE 3)
# Cols: 1 to 3
center_crop = grid[1:3, 1:3]

print("\n--- THE CROP (Rows 1-2, Cols 1-2) ---")
print(center_crop)