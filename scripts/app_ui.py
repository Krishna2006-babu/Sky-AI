import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import os
from datetime import datetime
from data_manager import DataManager
# --- IMPORT THE BRAIN ---
from skin_analysis import SkinAnalyzer 
from visualize_data import show_skin_progress  # <--- NEW IMPORT
class SkyAIApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1150x750")
        self.window.configure(bg="#2c3e50") 

        # --- INITIALIZE THE BRAIN ---
        self.analyzer = SkinAnalyzer()

        self.cap = None
        self.is_running = True

        # 1. VIDEO DISPLAY (Left)
        self.video_label = tk.Label(window, bg="black", width=640, height=480)
        self.video_label.grid(row=0, column=0, padx=20, pady=20)

        # 2. REPORT PANEL (Right)
        self.panel = tk.Frame(window, bg="#34495e", width=400, height=500)
        self.panel.grid(row=0, column=1, padx=20, pady=20, sticky="n")
        
        tk.Label(self.panel, text="SKIN REPORT", font=("Arial", 22, "bold"), fg="white", bg="#34495e").pack(pady=20)
        
        # VISUAL METERS
        self.oil_val_str = tk.StringVar(value="Oiliness: --")
        self.create_meter("OILINESS (Sebum %)", "oil_bar", self.oil_val_str)
        
        self.tex_val_str = tk.StringVar(value="Texture: --")
        self.create_meter("TEXTURE (Smoothness)", "tex_bar", self.tex_val_str)

        # NEW: ACNE METER
        self.acne_val_str = tk.StringVar(value="Acne: --")
        self.create_meter("ACNE (Inflammation)", "acne_bar", self.acne_val_str)

        self.status_label = tk.Label(self.panel, text="System: Initializing Cam...", font=("Arial", 10), fg="#f1c40f", bg="#34495e")
        self.status_label.pack(pady=10)

        # AI RECOMMENDATION BOX
        self.sol_frame = tk.Frame(self.panel, bg="#2c3e50", bd=1, relief="solid")
        self.sol_frame.pack(pady=10, padx=20, fill="both", expand=True)
        tk.Label(self.sol_frame, text="AI RECOMMENDATIONS", font=("Arial", 10, "bold"), fg="#38bdf8", bg="#2c3e50").pack(pady=5)
        self.sol_text = tk.Label(self.sol_frame, text="Scan to see results...", font=("Arial", 10), fg="white", bg="#2c3e50", wraplength=300, justify="left")
        self.sol_text.pack(pady=10, padx=10)

       

        # 3. SCAN BUTTON
        self.scan_btn = tk.Button(window, text="🚀 START AI SKIN SCAN", command=self.take_scan, bg="#27ae60", fg="white", font=("Arial", 14, "bold"), padx=80, pady=18)
        self.scan_btn.grid(row=1, column=0, columnspan=2, pady=10)

        self.start_camera_auto()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()
        # --- INITIALIZE THE BRAIN & MEMORY ---
        self.analyzer = SkinAnalyzer()
        self.db = DataManager() # <--- NEW LINE
        # SAVE REPORT BUTTON
        self.save_btn = tk.Button(self.panel, text="💾 SAVE REPORT", command=self.save_report, bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=20)
        self.save_btn.pack(pady=10)

        # --- NEW: GRAPH BUTTON ---
        self.graph_btn = tk.Button(self.panel, text="📈 VIEW TRENDS", command=show_skin_progress, bg="#9b59b6", fg="white", font=("Arial", 10, "bold"), padx=20)
        self.graph_btn.pack(pady=5)

    def create_meter(self, label_text, attr_name, str_var):
        frame = tk.Frame(self.panel, bg="#34495e")
        frame.pack(fill="x", padx=30)
        tk.Label(frame, text=label_text, fg="#bdc3c7", bg="#34495e", font=("Arial", 9, "bold")).pack(side="left")
        tk.Label(frame, textvariable=str_var, fg="white", bg="#34495e", font=("Arial", 9)).pack(side="right")
        bar = ttk.Progressbar(self.panel, orient="horizontal", length=320, mode="determinate")
        bar.pack(pady=(0, 15), padx=20)
        setattr(self, attr_name, bar)

    def start_camera_auto(self):
        threading.Thread(target=self.connect_hardware, args=(0,), daemon=True).start()

    def connect_hardware(self, index):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) 
        if self.cap.isOpened():
            self.status_label.config(text="✅ Internal Cam Active", fg="#2ecc71")
        else:
            self.status_label.config(text="❌ Cam Not Found", fg="#e74c3c")

    def update_frame(self):
        if self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    display_frame = cv2.resize(frame, (640, 480))
                    cv2_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2_img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
            
            self.window.after(15, self.update_frame)

    def take_scan(self):
        if not self.cap or not self.cap.isOpened(): 
            messagebox.showwarning("Warning", "Camera not active!")
            return
        
        ret, frame = self.cap.read()
        if ret:
            # 1. SEND RAW IMAGE TO BRAIN
            report = self.analyzer.analyze_image(frame)

            # 2. UPDATE UI METERS
            self.oil_bar['value'] = report['oil_score']
            self.oil_val_str.set(f"Oiliness: {report['oil_score']}%")
            
            self.tex_bar['value'] = report['texture_score']
            self.tex_val_str.set(f"Texture: {report['texture_score']}")

            # UPDATE NEW ACNE METER
            self.acne_bar['value'] = report['acne_score'] if 'acne_score' in report else 0
            self.acne_val_str.set(f"Acne: {report['acne_count']} spots")

            # 3. DISPLAY THE "MARKED UP" IMAGE
            marked_frame = report['annotated_image']
            marked_frame = cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB)
            marked_frame = cv2.resize(marked_frame, (640, 480))
            
            img = Image.fromarray(marked_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

           # 4. TEXT RESULTS
            issues_text = ", ".join(report['issues'])
            products_text = ", ".join(report['products'])
            
            # --- NEW: DISPLAY HEALTH SCORE ---
            score = report['overall_score']
            
            # Dynamic Color: Green for Good, Yellow for Average, Red for Bad
            if score > 80:
                score_color = "#2ecc71" # Green
            elif score > 50:
                score_color = "#f1c40f" # Yellow
            else:
                score_color = "#e74c3c" # Red
            
            # Update the Status Label to show the Score
            self.status_label.config(
                text=f"🏆 SKIN HEALTH SCORE: {score}/100", 
                fg=score_color, 
                font=("Arial", 16, "bold")
            )
            
            self.sol_text.config(text=f"ISSUES:\n{issues_text}\n\nSUGGESTED:\n{products_text}")
            
            self.is_running = False

    def save_report(self):
        # 1. Validation: Don't save empty data
        if "--" in self.oil_val_str.get():
            messagebox.showwarning("Warning", "Please run a scan before saving.")
            return
        
        # 2. Extract Numbers
        try:
            # Parse "Oiliness: 45.0%" -> 45.0
            oil = float(self.oil_val_str.get().split(": ")[1].replace("%", ""))
            
            # Parse "Texture: 25.0" -> 25.0
            tex = float(self.tex_val_str.get().split(": ")[1])
            
            # Parse "Acne: 12 spots" -> 12
            acne_count = int(self.acne_val_str.get().split(": ")[1].split(" ")[0])
            
            # Parse "🏆 SKIN HEALTH SCORE: 38/100" -> 38
            score_text = self.status_label.cget("text")
            if "SCORE:" in score_text:
                overall_score = int(score_text.split(": ")[1].split("/")[0])
            else:
                overall_score = 0 

            # 3. SAVE TO DATABASE
            success = self.db.log_scan(oil, tex, acne_count, overall_score)
            
            # --- THIS PART WAS MISSING ---
            if success:
                self.save_text_receipt() # Optional: keep saving text file if you want
                messagebox.showinfo("Success", "✅ Data logged to History Database!")
            else:
                messagebox.showerror("Error", "Failed to save to database.")
            # -----------------------------
                
        except Exception as e:
            print(f"Parsing Error: {e}")
            messagebox.showerror("Error", f"Could not parse data: {e}")
    def save_text_receipt(self):
        """Legacy text file saving (Keep this if you like the .txt receipts)"""
        if not os.path.exists("reports"):
            os.makedirs("reports")
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"reports/Scan_{now}.txt"
        with open(filename, "w") as f:
            f.write(f"SKY-AI SCAN\nDate: {datetime.now()}\n")
            f.write(f"{self.oil_val_str.get()}\n{self.tex_val_str.get()}\n{self.acne_val_str.get()}")
    def on_closing(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.window.destroy()