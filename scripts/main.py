import tkinter as tk
from app_ui import SkyAIApp  # Import the GUI class from your app_ui.py file

if __name__ == "__main__":
    # 1. Create the root window (The main container)
    root = tk.Tk()
    
    # 2. Launch the App
    # This initializes the SkyAIApp class, which sets up the camera and UI
    app = SkyAIApp(root, "Sky-AI Skin Analyzer")
    
    # 3. Start the Event Loop
    # This keeps the window open and listening for clicks
    root.mainloop()