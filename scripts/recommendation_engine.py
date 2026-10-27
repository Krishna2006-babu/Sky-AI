import pandas as pd
import os

class SkinRecommender:
    """
    The 'Pharmacist' of the Sky-AI project.
    It reads the CSV database and finds the best products based on the diagnosis.
    """
    def __init__(self, db_path="data/skin_products.csv"):
        self.db_path = db_path
        # Check if database exists
        if not os.path.exists(self.db_path):
            print(f"❌ Error: Database not found at {self.db_path}")
            self.knowledge_base = pd.DataFrame() # Empty backup
        else:
            # Load the CSV into a Pandas DataFrame
            self.knowledge_base = pd.read_csv(self.db_path)
            print("✅ Dermatologist Database Loaded Successfully")

    def get_advice(self, skin_type, concern):
        """
        Filters the database to find matches.
        """
        if self.knowledge_base.empty:
            return ["Database Error: No products found."]

        # PANDAS MAGIC: Filter data like SQL
        # We use a broad filter first: Match Skin Type AND Concern
        matches = self.knowledge_base[
            (self.knowledge_base['Skin_Type'] == skin_type) & 
            (self.knowledge_base['Concern'] == concern)
        ]
        
        # If no exact match, try matching just the Concern (fallback)
        if matches.empty:
            matches = self.knowledge_base[self.knowledge_base['Concern'] == concern]
            
        # Convert the results to a list of advice strings
        recommendations = []
        for index, row in matches.iterrows():
            rec = f"• {row['Ingredient']} ({row['Product_Type']}): {row['Action']}"
            recommendations.append(rec)
            
        if not recommendations:
            return ["No specific products found for this combination."]
            
        return recommendations

# --- TEST BLOCK (Run this file directly to test) ---
if __name__ == "__main__":
    rec = SkinRecommender()
    print("\n🧪 Test Case 1: Oily Skin with Acne")
    print("\n".join(rec.get_advice("Oily", "Acne")))