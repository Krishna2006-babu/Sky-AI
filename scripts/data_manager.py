import pandas as pd
import os
from datetime import datetime

class DataManager:
    """
    The 'Librarian' of Sky-AI.
    Updated for Week 4: Now tracks 'Overall_Score'.
    """
    def __init__(self, filename="data/skin_history.csv"):
        self.filename = filename
        
        # 1. Create folder if needed
        if not os.path.exists("data"):
            os.makedirs("data")
            
        # 2. Create CSV if it doesn't exist
        if not os.path.isfile(self.filename):
            print(f"📂 Creating new database at {self.filename}...")
            # Added 'Overall_Score' to columns
            df = pd.DataFrame(columns=["Date", "Time", "Oil_Score", "Texture_Score", "Acne_Count", "Overall_Score"])
            df.to_csv(self.filename, index=False)
        else:
            # 3. MIGRATION CHECK: If the file exists but lacks the new column, add it
            df = pd.read_csv(self.filename)
            if "Overall_Score" not in df.columns:
                print("⚠️ Upgrading database schema...")
                df["Overall_Score"] = 0 # Default value for old scans
                df.to_csv(self.filename, index=False)

    def log_scan(self, oil, texture, acne_count, overall_score):
        """
        Saves a single scan result.
        """
        now = datetime.now()
        
        new_data = {
            "Date": now.strftime("%Y-%m-%d"),
            "Time": now.strftime("%H:%M:%S"),
            "Oil_Score": round(oil, 1),
            "Texture_Score": round(texture, 1),
            "Acne_Count": int(acne_count),
            "Overall_Score": int(overall_score) # <--- SAVING THE NEW SCORE
        }
        
        try:
            df = pd.read_csv(self.filename)
            new_row = pd.DataFrame([new_data])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.filename, index=False)
            print("💾 Scan with Health Score saved successfully!")
            return True
        except Exception as e:
            print(f"❌ Error saving scan: {e}")
            return False