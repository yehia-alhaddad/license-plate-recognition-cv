"""
License Plate Recognition GUI Module (lPPR_gui.py renamed)
User interface for the integrated LPR system
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
import csv

# Import our renamed modules
import license_plate_detector  # Detection module (formerly IPPR2)
import ocr_processor          # OCR and post-processing module (formerly Part2)

# ---------------- TOOLTIP HELPER ----------------
def create_tooltip(widget, text):
    tooltip = tk.Toplevel(widget)
    tooltip.withdraw()
    tooltip.overrideredirect(True)
    label = tk.Label(tooltip, text=text, background="yellow", relief="solid", borderwidth=1)
    label.pack()

    def enter(event):
        tooltip.deiconify()
        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 20
        tooltip.geometry(f"+{x}+{y}")

    def leave(event):
        tooltip.withdraw()

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

# ---------------- MAIN GUI ----------------
class LPR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition System")
        self.root.geometry("1200x750")
        self.root.configure(bg="#e8f0f7")

        self.image_path = None
        self.cv_img = None
        self.results = []

        # ====== HEADER ======
        header = tk.Label(root, text="License Plate Recognition Dashboard",
                          font=("Helvetica", 22, "bold"),
                          bg="#2c3e50", fg="white", pady=15)
        header.pack(fill="x")

        # ====== MAIN CONTAINER ======
        main_frame = tk.Frame(root, bg="#e8f0f7")
        main_frame.pack(fill="both", expand=True)

        # LEFT SIDEBAR
        sidebar = tk.Frame(main_frame, width=250, bg="#34495e")
        sidebar.pack(side="left", fill="y")

        upload_btn = tk.Button(sidebar, text="Upload Image", command=self.upload_image,
                               font=("Arial", 12), bg="#1abc9c", fg="white", width=20)
        upload_btn.pack(pady=20)
        create_tooltip(upload_btn, "Select a vehicle image for processing")

        self.detect_btn = tk.Button(sidebar, text="Detect Plate", command=self.detect_plate,
                                    font=("Arial", 12), bg="#2980b9", fg="white", width=20, state="disabled")
        self.detect_btn.pack(pady=10)
        create_tooltip(self.detect_btn, "Run license plate detection on the uploaded image")

        self.export_btn = tk.Button(sidebar, text="Export Results", command=self.export_csv,
                                    font=("Arial", 12), bg="#f39c12", fg="white", width=20, state="disabled")
        self.export_btn.pack(pady=10)
        create_tooltip(self.export_btn, "Save detection results as a CSV file")

        self.reset_btn = tk.Button(sidebar, text="Reset", command=self.reset_gui,
                                   font=("Arial", 12), bg="#8e44ad", fg="white", width=20)
        self.reset_btn.pack(pady=10)
        create_tooltip(self.reset_btn, "Clear image and results to start fresh")

        exit_btn = tk.Button(sidebar, text="Exit", command=root.quit,
                             font=("Arial", 12), bg="#c0392b", fg="white", width=20)
        exit_btn.pack(pady=50)
        create_tooltip(exit_btn, "Close the application")

        # RIGHT CONTENT
        content = tk.Frame(main_frame, bg="white")
        content.pack(side="right", expand=True, fill="both")

        # IMAGE DISPLAY inside frame
        img_frame = tk.Frame(content, bg="#bdc3c7", padx=5, pady=5)
        img_frame.pack(pady=10)
        self.lbl_img = tk.Label(img_frame, bg="white")
        self.lbl_img.pack()

        # RESULTS TABLE + scrollbar
        columns = ("Image", "Plate", "State", "Confidence")
        self.table = ttk.Treeview(content, columns=columns, show="headings", height=8)
        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, anchor="center", width=150)
        self.table.pack(side="left", pady=20, padx=10, fill="both", expand=True)

        scroll_y = tk.Scrollbar(content, orient="vertical", command=self.table.yview)
        self.table.configure(yscroll=scroll_y.set)
        scroll_y.pack(side="right", fill="y")

        # Table styling
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background="#f9f9f9",
                        foreground="black",
                        rowheight=30,
                        fieldbackground="#f9f9f9")
        style.configure("Treeview.Heading",
                        font=("Helvetica", 12, "bold"),
                        background="#2c3e50",
                        foreground="white")
        style.map("Treeview", background=[("selected", "#3498db")])

        # STATUS BAR
        self.status = tk.Label(root, text="Ready", bd=1, relief="sunken",
                               anchor="w", bg="white", fg="black")
        self.status.pack(side="bottom", fill="x")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        self.image_path = file_path
        self.cv_img = cv2.imread(file_path)

        # Show preview
        disp_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        disp_img = cv2.resize(disp_img, (700, 500))
        disp_img = Image.fromarray(disp_img)
        photo = ImageTk.PhotoImage(disp_img)
        self.lbl_img.config(image=photo)
        self.lbl_img.image = photo

        self.detect_btn.config(state="normal")
        self.export_btn.config(state="disabled")
        self.status.config(text=f"Loaded {os.path.basename(file_path)}")

    def detect_plate(self):
        """Use license_plate_detector for detection and ocr_processor for enhanced OCR processing"""
        if self.cv_img is None:
            messagebox.showerror("Error", "No image uploaded!")
            return

        try:
            # Use license_plate_detector for detection
            best_box, best_text, best_state, best_conf = license_plate_detector.process_image(self.cv_img)
            
            if best_box:
                try:
                    # Enhance OCR result using ocr_processor
                    enhanced_text, enhanced_state, enhanced_conf = ocr_processor.enhance_ocr_result(best_text, best_conf)
                    
                    # Use enhanced results if they're better
                    if enhanced_conf > best_conf:
                        best_text, best_state, best_conf = enhanced_text, enhanced_state, enhanced_conf
                except Exception as e:
                    # If OCR enhancement fails, use basic results
                    print(f"OCR enhancement failed: {e}")
                    # Continue with basic results
                
                # Draw detection box on image
                x, y, w, h = best_box
                orig = self.cv_img.copy()
                cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(orig, f"{best_text} | {best_state}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Display processed image
                disp_img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                disp_img = cv2.resize(disp_img, (700, 500))
                disp_img = Image.fromarray(disp_img)
                photo = ImageTk.PhotoImage(disp_img)
                self.lbl_img.config(image=photo)
                self.lbl_img.image = photo

                # Add to results table
                self.table.insert("", "end", values=(
                    os.path.basename(self.image_path),
                    best_text,
                    best_state,
                    f"{best_conf:.2f}"
                ))

                self.results.append((os.path.basename(self.image_path),
                                     best_text, best_state, best_conf))
                self.export_btn.config(state="normal")
                self.status.config(text=f"Detection successful: {best_text} | {best_state}")
            else:
                messagebox.showwarning("Detection", "No plate detected")
                self.status.config(text="Detection failed")
                
        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            if "tesseract" in str(e).lower():
                error_msg += "\n\nTesseract OCR is not available. Please install it for full functionality."
                error_msg += "\nDownload from: https://github.com/UB-Mannheim/tesseract/wiki"
            messagebox.showerror("Detection Error", error_msg)
            self.status.config(text="Detection error occurred")

    def export_csv(self):
        if not self.results:
            messagebox.showerror("Error", "No results to export!")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV Files", "*.csv")])
        if not save_path:
            return

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "Plate", "State", "Confidence"])
            writer.writerows(self.results)

        messagebox.showinfo("Success", f"Results exported to {save_path}")
        self.status.config(text=f"Results saved: {os.path.basename(save_path)}")

    def reset_gui(self):
        """Reset everything back to initial state."""
        self.image_path = None
        self.cv_img = None
        self.results.clear()

        # Clear image preview
        self.lbl_img.config(image="", text="")

        # Clear table
        for row in self.table.get_children():
            self.table.delete(row)

        # Reset buttons
        self.detect_btn.config(state="disabled")
        self.export_btn.config(state="disabled")

        self.status.config(text="Interface reset, ready for new image")
        messagebox.showinfo("Reset", "Interface has been reset")

# Run app
if __name__ == "__main__":
    root = tk.Tk()
    app = LPR_GUI(root)
    root.mainloop()