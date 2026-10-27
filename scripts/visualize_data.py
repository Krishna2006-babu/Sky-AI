import pandas as pd
import matplotlib.pyplot as plt
import os

def show_skin_progress():
    filename = "data/skin_history.csv"
    
    # 1. Check if data exists
    if not os.path.exists(filename):
        print("❌ No data found! Run the app and save a scan first.")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(filename)
        
        # Combine Date & Time into a single "Timestamp" column for the X-axis
        df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        
        # Sort by time just in case
        df = df.sort_values('Timestamp')

        # 3. Setup the Graph
        plt.figure(figsize=(12, 6)) # Width: 12 inches, Height: 6 inches
        plt.style.use('ggplot')     # Make it look professional
        
        # --- LINE 1: Overall Health Score (The most important one) ---
        plt.plot(df['Timestamp'], df['Overall_Score'], 
                 label='🏆 Health Score', color='green', linewidth=3, marker='o')

        # --- LINE 2: Oiliness (Dashed line) ---
        plt.plot(df['Timestamp'], df['Oil_Score'], 
                 label='💧 Oiliness', color='blue', linestyle='--', alpha=0.6)

        # --- LINE 3: Texture (Dotted line) ---
        plt.plot(df['Timestamp'], df['Texture_Score'], 
                 label='🧩 Texture', color='orange', linestyle=':', alpha=0.6)

        # 4. Labeling
        plt.title('My Skin Health Journey', fontsize=16)
        plt.xlabel('Date & Time')
        plt.ylabel('Score (0-100)')
        plt.ylim(0, 105) # Fixed scale so 100 is always at the top
        plt.legend()     # Show the key
        plt.grid(True)
        
        # Rotate dates so they don't overlap
        plt.xticks(rotation=45)
        
        # 5. Show it!
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error graphing data: {e}")

# Allow running this file directly to test
if __name__ == "__main__":
    show_skin_progress()