# import antigravity
import os
import cv2
import threading
import logging
import queue
import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd

from stitcher import UAVStitcher
from analyzer import AssetAnalyzer

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
    def emit(self, record):
        self.log_queue.put(self.format(record))

logger = logging.getLogger("UAVEngine")
logger.setLevel(logging.INFO)

class UAVApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("UAV Intelligence Engine (Research Edition)")
        self.geometry("900x700") 
        
        self.video_path = None
        self.stitched_map = None
        self.analyzed_map = None
        self.report_df = None
        self.asset_count = 0
        
        self.log_queue = queue.Queue()
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        q_handler = QueueHandler(self.log_queue)
        q_handler.setFormatter(formatter)
        logger.addHandler(q_handler)

        self.create_widgets()
        self.after(100, self.poll_logs)

    def poll_logs(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.console.configure(state="normal")
            self.console.insert("end", msg + "\n")
            self.console.see("end")
            self.console.configure(state="disabled")
        self.after(100, self.poll_logs)

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.tabview.add("Pipeline Configuration")
        self.tabview.add("Advanced Settings")
        self.tabview.add("System Console")

        self._build_config_tab()
        self._build_advanced_tab()
        self._build_console_tab()
        
        # Bottom Control Panel
        self.frame_bottom = ctk.CTkFrame(self)
        self.frame_bottom.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.frame_bottom.grid_columnconfigure(1, weight=1)
        
        self.btn_run = ctk.CTkButton(self.frame_bottom, text="▶ INITIATE PIPELINE", font=("Arial", 14, "bold"), height=40, command=self.run_pipeline_thread)
        self.btn_run.grid(row=0, column=0, padx=20, pady=20)
        
        self.progress_bar = ctk.CTkProgressBar(self.frame_bottom)
        self.progress_bar.grid(row=0, column=1, padx=20, pady=20, sticky="ew")
        self.progress_bar.set(0)
        
        self.btn_export_map = ctk.CTkButton(self.frame_bottom, text="Export Map (.png)", command=self.export_map, state="disabled")
        self.btn_export_map.grid(row=0, column=2, padx=10, pady=20)

        self.btn_export_csv = ctk.CTkButton(self.frame_bottom, text="Export Data (.csv)", command=self.export_csv, state="disabled")
        self.btn_export_csv.grid(row=0, column=3, padx=10, pady=20)

    def _build_config_tab(self):
        tab = self.tabview.tab("Pipeline Configuration")
        tab.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(tab, text="1. Input Source", font=("Arial", 16, "bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.btn_select_video = ctk.CTkButton(tab, text="Browse Video", command=self.select_video)
        self.btn_select_video.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.lbl_video_path = ctk.CTkLabel(tab, text="No video selected", text_color="gray")
        self.lbl_video_path.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        ctk.CTkLabel(tab, text="2. Detection Engine", font=("Arial", 16, "bold")).grid(row=2, column=0, padx=10, pady=(20,10), sticky="w")
        self.mode_var = ctk.StringVar(value="ai")
        self.radio_ai = ctk.CTkRadioButton(tab, text="Deep Learning (YOLOv8 Objects)", variable=self.mode_var, value="ai")
        self.radio_ai.grid(row=3, column=0, padx=20, pady=10, sticky="w")
        self.radio_hsv = ctk.CTkRadioButton(tab, text="Spectral Tracking (HSV Colors)", variable=self.mode_var, value="hsv")
        self.radio_hsv.grid(row=4, column=0, padx=20, pady=10, sticky="w")

        ctk.CTkLabel(tab, text="3. Telemetry", font=("Arial", 16, "bold")).grid(row=5, column=0, padx=10, pady=(20,10), sticky="w")
        self.lbl_status = ctk.CTkLabel(tab, text="Status: IDLE", font=("Courier", 14))
        self.lbl_status.grid(row=6, column=0, padx=20, pady=5, sticky="w")
        self.lbl_result = ctk.CTkLabel(tab, text="Total Assets: 0", font=("Courier", 16, "bold"), text_color="#2ecc71")
        self.lbl_result.grid(row=7, column=0, padx=20, pady=5, sticky="w")

    def _build_advanced_tab(self):
        tab = self.tabview.tab("Advanced Settings")
        
        ctk.CTkLabel(tab, text="Spectral Analysis Config", font=("Arial", 14, "bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        frame_hsv = ctk.CTkFrame(tab)
        frame_hsv.grid(row=1, column=0, padx=20, pady=5, sticky="w")
        ctk.CTkLabel(frame_hsv, text="HSV Lower Limit:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_hsv_lower = ctk.CTkEntry(frame_hsv, width=120)
        self.entry_hsv_lower.insert(0, "35, 100, 100")
        self.entry_hsv_lower.grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(frame_hsv, text="HSV Upper Limit:").grid(row=1, column=0, padx=5, pady=5)
        self.entry_hsv_upper = ctk.CTkEntry(frame_hsv, width=120)
        self.entry_hsv_upper.insert(0, "85, 255, 255")
        self.entry_hsv_upper.grid(row=1, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(tab, text="Deep Learning Config", font=("Arial", 14, "bold")).grid(row=2, column=0, padx=10, pady=(20,10), sticky="w")
        frame_ai = ctk.CTkFrame(tab)
        frame_ai.grid(row=3, column=0, padx=20, pady=5, sticky="w")
        ctk.CTkLabel(frame_ai, text="Confidence Threshold:").grid(row=0, column=0, padx=5, pady=5)
        self.slider_conf = ctk.CTkSlider(frame_ai, from_=0.1, to=0.9, number_of_steps=80)
        self.slider_conf.set(0.3)
        self.slider_conf.grid(row=0, column=1, padx=10, pady=5)
        
        ctk.CTkLabel(tab, text="Geospatial Estimation", font=("Arial", 14, "bold")).grid(row=4, column=0, padx=10, pady=(20,10), sticky="w")
        frame_geo = ctk.CTkFrame(tab)
        frame_geo.grid(row=5, column=0, padx=20, pady=5, sticky="w")
        ctk.CTkLabel(frame_geo, text="Downsample Width (px):").grid(row=0, column=0, padx=5, pady=5)
        self.entry_downscale = ctk.CTkEntry(frame_geo, width=100)
        self.entry_downscale.insert(0, "1280")
        self.entry_downscale.grid(row=0, column=1, padx=5, pady=5)

    def _build_console_tab(self):
        tab = self.tabview.tab("System Console")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        self.console = ctk.CTkTextbox(tab, font=("Courier", 12), fg_color="#1e1e1e", text_color="#00ff00")
        self.console.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.console.configure(state="disabled")

    def select_video(self):
        filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
        if filepath:
            self.video_path = filepath
            self.lbl_video_path.configure(text=os.path.basename(filepath))
            logger.info(f"Target locked: {filepath}")

    def update_progress(self, val):
        self.progress_bar.set(val)
        self.update_idletasks()

    def run_pipeline_thread(self):
        if not self.video_path:
            logger.error("Operation halted: No input source provided.")
            messagebox.showerror("Error", "Please select a video feed.")
            return

        mode = self.mode_var.get()
        hsv_lower, hsv_upper = None, None
        
        if mode == "hsv":
            try:
                hsv_lower = tuple(map(int, self.entry_hsv_lower.get().split(',')))
                hsv_upper = tuple(map(int, self.entry_hsv_upper.get().split(',')))
            except ValueError:
                logger.error("Invalid HSV parameters.")
                messagebox.showerror("Error", "Invalid HSV values.")
                return

        conf = float(self.slider_conf.get())
        downscale = int(self.entry_downscale.get())

        self.btn_run.configure(state="disabled")
        self.btn_export_map.configure(state="disabled")
        self.btn_export_csv.configure(state="disabled")
        
        self.progress_bar.set(0)
        self.lbl_status.configure(text="Status: STITCHING", text_color="yellow")
        
        # Switch to console tab automatically for cool effect
        self.tabview.set("System Console")
        logger.info("===================================")
        logger.info(f"INITIATING PIPELINE: MODE = {mode.upper()}")

        threading.Thread(target=self.run_pipeline, args=(mode, hsv_lower, hsv_upper, conf, downscale), daemon=True).start()

    def run_pipeline(self, mode, hsv_lower, hsv_upper, conf, downscale):
        try:
            stitcher = UAVStitcher(fps_sample=2, downscale_width=downscale)
            self.stitched_map = stitcher.stitch(self.video_path, self.update_progress)

            if self.stitched_map is None:
                self.after(0, self.pipeline_failed, "Feature extraction failed. Could not compute homography.")
                return

            self.after(0, self.lbl_status.configure, {"text": "Status: ANALYZING", "text_color": "orange"})

            analyzer = AssetAnalyzer(mode=mode, hsv_lower=hsv_lower, hsv_upper=hsv_upper, min_area=150, conf_threshold=conf)
            self.asset_count, self.analyzed_map, self.report_df = analyzer.analyze(self.stitched_map)

            self.after(0, self.pipeline_success)
        except Exception as e:
            logger.exception("Fatal error in pipeline.")
            self.after(0, self.pipeline_failed, str(e))

    def pipeline_success(self):
        self.lbl_status.configure(text="Status: COMPLETE", text_color="#2ecc71")
        self.lbl_result.configure(text=f"Total Assets: {self.asset_count}")
        self.btn_run.configure(state="normal")
        self.btn_export_map.configure(state="normal")
        
        if self.report_df is not None and not self.report_df.empty:
            self.btn_export_csv.configure(state="normal")
            
        self.progress_bar.set(1.0)
        self.tabview.set("Pipeline Configuration")
        logger.info(f"Pipeline executed successfully. {self.asset_count} target(s) locked.")
        messagebox.showinfo("Mission Success", f"Analysis complete. Discovered {self.asset_count} assets.")

    def pipeline_failed(self, error_msg):
        self.lbl_status.configure(text="Status: FAILED", text_color="red")
        self.btn_run.configure(state="normal")
        self.progress_bar.set(0)
        messagebox.showerror("System Failure", error_msg)

    def get_output_dir(self, subfolder):
        import sys
        if getattr(sys, 'frozen', False):
            # If running from dist/main.exe, root is one level up
            root_dir = os.path.abspath(os.path.join(os.path.dirname(sys.executable), ".."))
        else:
            # If running from src/main.py, root is one level up
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            
        out_dir = os.path.join(root_dir, "output", subfolder)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def export_map(self):
        if self.analyzed_map is None: return
        initial_dir = self.get_output_dir("maps")
        filepath = filedialog.asksaveasfilename(initialdir=initial_dir, defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if filepath:
            cv2.imwrite(filepath, self.analyzed_map)
            logger.info(f"Topographical map exported to {filepath}")
            messagebox.showinfo("Export Success", "Map successfully saved.")

    def export_csv(self):
        if self.report_df is None or self.report_df.empty: return
        initial_dir = self.get_output_dir("csv")
        filepath = filedialog.asksaveasfilename(initialdir=initial_dir, defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.report_df.to_csv(filepath, index=False)
            logger.info(f"Analytics report exported to {filepath}")
            messagebox.showinfo("Export Success", "Analytics report successfully generated.")

if __name__ == "__main__":
    app = UAVApp()
    app.mainloop()
