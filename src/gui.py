"""
GUI interface for MP4 to Lottie converter using Tkinter.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from typing import Optional, Callable
import logging

from .video_processor import VideoProcessor
from .lottie_converter import LottieConverter
from .utils import FileUtils, ValidationUtils, ProgressTracker, ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MP4ToLottieGUI:
    """Main GUI application for MP4 to Lottie conversion."""

    def __init__(self):
        """Initialize the GUI application."""
        self.root = tk.Tk()
        self.root.title("MP4 to Lottie Converter")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # Configuration
        self.config = ConfigManager()

        # State variables
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.path.expanduser("~/Desktop"))
        self.max_frames = tk.IntVar(value=self.config.get("max_frames", 150))
        self.quality = tk.IntVar(value=self.config.get("output_quality", 80))
        self.max_file_size = tk.DoubleVar(
            value=self.config.get("max_file_size_mb", 5.0)
        )

        # Processing state
        self.is_processing = False
        self.current_processor = None
        self.progress_tracker = None

        self.setup_ui()
        self.setup_styles()

    def setup_styles(self):
        """Setup custom styles for the GUI."""
        style = ttk.Style()

        # Configure styles
        style.configure("Title.TLabel", font=("Arial", 16, "bold"))
        style.configure("Heading.TLabel", font=("Arial", 12, "bold"))
        style.configure("Success.TLabel", foreground="green")
        style.configure("Error.TLabel", foreground="red")
        style.configure("Warning.TLabel", foreground="orange")

    def setup_ui(self):
        """Setup the user interface components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame, text="MP4 to Lottie Converter", style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Input file selection
        self.setup_file_selection(main_frame, row=1)

        # Output directory selection
        self.setup_output_selection(main_frame, row=2)

        # Conversion settings
        self.setup_settings(main_frame, row=3)

        # Video information display
        self.setup_video_info(main_frame, row=4)

        # Progress section
        self.setup_progress_section(main_frame, row=5)

        # Control buttons
        self.setup_control_buttons(main_frame, row=6)

        # Status bar
        self.setup_status_bar(main_frame, row=7)

    def setup_file_selection(self, parent, row):
        """Setup file selection section."""
        # Input file frame
        file_frame = ttk.LabelFrame(parent, text="Input Video File", padding="10")
        file_frame.grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="MP4 File:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )

        self.file_entry = ttk.Entry(
            file_frame, textvariable=self.input_file, state="readonly"
        )
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        self.browse_button = ttk.Button(
            file_frame, text="Browse...", command=self.browse_input_file
        )
        self.browse_button.grid(row=0, column=2, sticky=tk.W)

    def setup_output_selection(self, parent, row):
        """Setup output directory selection."""
        output_frame = ttk.LabelFrame(parent, text="Output Directory", padding="10")
        output_frame.grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Directory:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )

        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir)
        self.output_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        self.output_browse_button = ttk.Button(
            output_frame, text="Browse...", command=self.browse_output_dir
        )
        self.output_browse_button.grid(row=0, column=2, sticky=tk.W)

    def setup_settings(self, parent, row):
        """Setup conversion settings."""
        settings_frame = ttk.LabelFrame(
            parent, text="Conversion Settings", padding="10"
        )
        settings_frame.grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        settings_frame.columnconfigure(1, weight=1)

        # Max frames setting
        ttk.Label(settings_frame, text="Max Frames:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )
        self.frames_spinbox = ttk.Spinbox(
            settings_frame, from_=10, to=1000, textvariable=self.max_frames, width=10
        )
        self.frames_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))

        # Quality setting
        ttk.Label(settings_frame, text="Quality (%):").grid(
            row=0, column=2, sticky=tk.W, padx=(0, 10)
        )
        self.quality_spinbox = ttk.Spinbox(
            settings_frame, from_=10, to=100, textvariable=self.quality, width=10
        )
        self.quality_spinbox.grid(row=0, column=3, sticky=tk.W)

        # Max file size setting
        ttk.Label(settings_frame, text="Max Size (MB):").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0)
        )
        self.size_spinbox = ttk.Spinbox(
            settings_frame,
            from_=1.0,
            to=100.0,
            increment=0.5,
            textvariable=self.max_file_size,
            width=10,
            format="%.1f",
        )
        self.size_spinbox.grid(row=1, column=1, sticky=tk.W, padx=(0, 20), pady=(10, 0))

    def setup_video_info(self, parent, row):
        """Setup video information display."""
        info_frame = ttk.LabelFrame(parent, text="Video Information", padding="10")
        info_frame.grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar for info text
        info_scrollbar = ttk.Scrollbar(
            info_frame, orient=tk.VERTICAL, command=self.info_text.yview
        )
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=info_scrollbar.set)

        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)

    def setup_progress_section(self, parent, row):
        """Setup progress display section."""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        progress_frame.columnconfigure(0, weight=1)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # Progress label
        self.progress_label = ttk.Label(progress_frame, text="Ready to convert")
        self.progress_label.grid(row=1, column=0, sticky=tk.W)

    def setup_control_buttons(self, parent, row):
        """Setup control buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, columnspan=3, pady=(10, 0))

        self.convert_button = ttk.Button(
            button_frame,
            text="Convert to Lottie",
            command=self.start_conversion,
            style="Accent.TButton",
        )
        self.convert_button.pack(side=tk.LEFT, padx=(0, 10))

        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_conversion,
            state=tk.DISABLED,
        )
        self.cancel_button.pack(side=tk.LEFT, padx=(0, 10))

        self.preview_button = ttk.Button(
            button_frame,
            text="Preview Result",
            command=self.preview_result,
            state=tk.DISABLED,
        )
        self.preview_button.pack(side=tk.LEFT)

    def setup_status_bar(self, parent, row):
        """Setup status bar."""
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)
        )

    def browse_input_file(self):
        """Browse for input MP4 file."""
        file_path = filedialog.askopenfilename(
            title="Select MP4 Video File",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
        )

        if file_path:
            if FileUtils.validate_video_file(file_path):
                self.input_file.set(file_path)
                self.load_video_info()
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            else:
                messagebox.showerror(
                    "Invalid File", "Please select a valid MP4 video file."
                )

    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)

    def load_video_info(self):
        """Load and display video information."""
        if not self.input_file.get():
            return

        try:
            processor = VideoProcessor(self.input_file.get())
            info = processor.get_video_info()

            # Try motion analysis with error handling
            try:
                motion_info = processor.analyze_motion()
                motion_text = f"""Motion Analysis:
Motion Detected: {'Yes' if motion_info['motion_detected'] else 'No'}
Average Motion: {motion_info['average_motion']:.2f}
Analysis Method: {'Optical Flow' if motion_info['motion_samples'] > 0 else 'Frame Difference'}"""
            except Exception as motion_error:
                logger.warning(f"Motion analysis failed: {motion_error}")
                motion_text = """Motion Analysis:
Motion Detected: Unknown (analysis failed)
Average Motion: N/A
Note: Motion analysis encountered an error but video processing will continue."""

            info_text = f"""File: {os.path.basename(info['path'])}
Dimensions: {info['width']} x {info['height']} pixels
Duration: {info['duration']:.2f} seconds
Frame Rate: {info['fps']:.2f} FPS
Total Frames: {info['frame_count']}
Aspect Ratio: {info['aspect_ratio']:.2f}

{motion_text}

Estimated Output Size: {ValidationUtils.estimate_output_size(
    min(self.max_frames.get(), info['frame_count']), 
    info['width'], 
    info['height'], 
    self.quality.get()
):.2f} MB"""

            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info_text)
            self.info_text.config(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Error loading video info: {e}")

            # Show a more user-friendly error message
            error_msg = "Failed to load video information.\n\n"
            if "OpenCV" in str(e):
                error_msg += "This appears to be a video processing error. The video file may be:\n"
                error_msg += "• Corrupted or incomplete\n"
                error_msg += "• In an unsupported format\n"
                error_msg += "• Too large or complex\n\n"
                error_msg += "Try using a different MP4 file or convert your video to a standard MP4 format."
            else:
                error_msg += f"Error details: {str(e)}"

            messagebox.showerror("Video Loading Error", error_msg)

    def start_conversion(self):
        """Start the conversion process in a separate thread."""
        if not self.input_file.get():
            messagebox.showerror("No Input", "Please select an MP4 file first.")
            return

        if not os.path.exists(self.output_dir.get()):
            messagebox.showerror(
                "Invalid Output", "Please select a valid output directory."
            )
            return

        # Disable UI during processing
        self.set_processing_state(True)

        # Start conversion in separate thread
        conversion_thread = threading.Thread(
            target=self.perform_conversion, daemon=True
        )
        conversion_thread.start()

    def perform_conversion(self):
        """Perform the actual conversion process."""
        try:
            # Initialize processor
            self.current_processor = VideoProcessor(self.input_file.get())
            video_info = self.current_processor.get_video_info()

            # Create output directory
            video_name = os.path.splitext(os.path.basename(self.input_file.get()))[0]
            output_dir = FileUtils.create_output_directory(
                self.output_dir.get(), video_name
            )
            frames_dir = os.path.join(output_dir, "frames")

            # Setup progress tracking
            total_steps = 3  # Extract frames, convert to Lottie, save file
            self.progress_tracker = ProgressTracker(total_steps, "Converting")
            self.progress_tracker.add_callback(self.update_progress)

            # Step 1: Extract frames
            self.update_status("Extracting frames from video...")
            frame_paths = self.current_processor.extract_frames(
                frames_dir, self.max_frames.get()
            )
            self.progress_tracker.update(1, f"Extracted {len(frame_paths)} frames")

            if not frame_paths:
                raise ValueError("No frames could be extracted from the video")

            # Step 2: Convert to Lottie
            self.update_status("Converting frames to Lottie format...")
            converter = LottieConverter(
                video_info["width"], video_info["height"], video_info["fps"]
            )

            # Use optimized conversion if file size limit is set
            if self.max_file_size.get() > 0:
                lottie_data = converter.create_optimized_animation(
                    frame_paths, self.max_file_size.get(), video_info["duration"]
                )
            else:
                lottie_data = converter.create_sequence_animation(
                    frame_paths, original_duration=video_info["duration"]
                )

            self.progress_tracker.update(2, "Generated Lottie animation")

            # Step 3: Save file
            self.update_status("Saving Lottie file...")
            output_file = os.path.join(output_dir, "output", f"{video_name}.json")

            if converter.save_animation(lottie_data, output_file):
                self.progress_tracker.update(3, "Conversion complete!")

                # Validate the output
                if converter.validate_lottie_format(lottie_data):
                    file_size = FileUtils.get_file_size_mb(output_file)
                    self.show_success_message(output_file, file_size)
                else:
                    self.show_error_message(
                        "Generated file may not be valid Lottie format"
                    )
            else:
                raise ValueError("Failed to save Lottie file")

            # Clean up temporary files if configured
            if self.config.get("temp_cleanup", True):
                FileUtils.clean_temp_files(frames_dir)

        except Exception as e:
            logger.error(f"Conversion error: {e}")
            self.show_error_message(str(e))

        finally:
            # Re-enable UI
            self.root.after(0, lambda: self.set_processing_state(False))

    def cancel_conversion(self):
        """Cancel the current conversion process."""
        # Note: This is a simplified cancel - in a production app you'd want
        # more sophisticated thread cancellation
        self.is_processing = False
        self.update_status("Conversion cancelled")
        self.set_processing_state(False)

    def preview_result(self):
        """Open the result file for preview."""
        # This would typically open the Lottie file in a preview application
        messagebox.showinfo(
            "Preview",
            "Preview functionality would open the Lottie file in a compatible viewer.",
        )

    def update_progress(
        self, current: int, total: int, percentage: float, message: str
    ):
        """Update progress bar and label."""

        def update_ui():
            self.progress_var.set(percentage)
            self.progress_label.config(text=message)

        self.root.after(0, update_ui)

    def update_status(self, message: str):
        """Update status bar message."""

        def update_ui():
            self.status_var.set(message)

        self.root.after(0, update_ui)

    def show_success_message(self, output_file: str, file_size: float):
        """Show success message."""

        def show_message():
            message = f"Conversion completed successfully!\n\nOutput file: {output_file}\nFile size: {file_size:.2f} MB"
            messagebox.showinfo("Success", message)
            self.preview_button.config(state=tk.NORMAL)

        self.root.after(0, show_message)

    def show_error_message(self, error: str):
        """Show error message."""

        def show_message():
            messagebox.showerror("Conversion Error", f"Conversion failed:\n\n{error}")

        self.root.after(0, show_message)

    def set_processing_state(self, processing: bool):
        """Enable/disable UI elements during processing."""
        self.is_processing = processing

        state = tk.DISABLED if processing else tk.NORMAL
        cancel_state = tk.NORMAL if processing else tk.DISABLED

        self.browse_button.config(state=state)
        self.output_browse_button.config(state=state)
        self.convert_button.config(state=state)
        self.cancel_button.config(state=cancel_state)
        self.frames_spinbox.config(state=state)
        self.quality_spinbox.config(state=state)
        self.size_spinbox.config(state=state)

        if not processing:
            self.progress_var.set(0)
            self.progress_label.config(text="Ready to convert")
            self.update_status("Ready")

    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
            messagebox.showerror(
                "Application Error", f"An unexpected error occurred: {e}"
            )


def main():
    """Main entry point for the GUI application."""
    app = MP4ToLottieGUI()
    app.run()


if __name__ == "__main__":
    main()
