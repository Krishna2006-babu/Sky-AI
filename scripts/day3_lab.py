# --- LAB EXPERIMENT: LISTS ---

# 1. Create a LIST of skin types
# Remember: We use square brackets [] for lists
skin_types = ["Oily", "Dry", "Normal", "Sensitive"]

print("All types:" , skin_types)\

# 3. Access specific items (INDEXING)
# Challenge: What will index 0 and 2 print?
print("First Item (Index 0):", skin_types[0])
print("Third Item (Index 2):", skin_types[2])

# 4. Modifying the list (Change "Dry" to "Dehydrated")
print("--- Updating List ---")
skin_types[1] = "Dehydrated" 
print("New List:", skin_types)