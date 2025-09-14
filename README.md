# MP4 to Lottie Converter

Convert MP4 videos to Lottie JSON animations with high quality and small file sizes (2-5MB).

## 🚀 Quick Start

### Step 1: Install Python

1. Go to [python.org](https://www.python.org/downloads/)
2. Download Python 3.8 or newer
3. Install Python (make sure to check "Add to PATH")

### Step 2: Setup Project

**Option A: Clone from GitHub**

```bash
# HTTPS (recommended for most users)
git clone https://github.com/azhar736/MP4-to-Lottie-Converter.git
cd MP4-to-Lottie-Converter

# SSH (if you have SSH keys set up)
git clone git@github.com:azhar736/MP4-to-Lottie-Converter.git
cd MP4-to-Lottie-Converter
```

**Option B: Download ZIP**

1. Download ZIP from [GitHub repository](https://github.com/azhar736/MP4-to-Lottie-Converter)
2. Extract the ZIP file
3. Open terminal/command prompt in project folder

### Step 3: Choose Your Method

#### Method A: Easy Setup (Recommended)

```bash
# Run setup script (creates virtual environment automatically)
python setup_venv.py

# Run the converter
./run_venv.sh          # Mac/Linux
run_venv.bat           # Windows
```

#### Method B: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Run the converter
python main.py
```

## 📱 How to Use

### GUI Mode (Easy)

1. Run `./run_venv.sh` (Mac/Linux) or `run_venv.bat` (Windows)
2. Click "Browse" to select your MP4 video
3. Choose output folder
4. Adjust settings:
   - **Quality**: 80-85% for best quality, 70-75% for smaller files
   - **Max Frames**: 150-200 for smooth animation
   - **Max Size**: 5MB (recommended)
5. Click "Convert to Lottie"

### Command Line Mode

```bash
# Basic conversion
./run_venv.sh video.mp4

# High quality (may be larger)
./run_venv.sh video.mp4 --quality 85 --max-frames 200

# Smaller size
./run_venv.sh video.mp4 --quality 75 --max-frames 120 --max-size 3.0

# Specify output folder
./run_venv.sh video.mp4 -o /path/to/output
```

## ⚙️ Settings Guide

| Setting        | Recommended | Effect                               |
| -------------- | ----------- | ------------------------------------ |
| **Quality**    | 80%         | Higher = better quality, larger file |
| **Max Frames** | 150         | More frames = smoother animation     |
| **Max Size**   | 5MB         | Target file size limit               |

### Quality vs Size Examples

- **Maximum Quality**: Quality 85%, Frames 200 → 4-5MB, excellent quality
- **Balanced**: Quality 80%, Frames 150 → 3-4MB, great quality
- **Compact**: Quality 75%, Frames 120 → 2-3MB, good quality

## 📁 Project Structure

```
MP4-to-Lottie-Converter/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── setup_venv.py       # Easy setup script
├── run_venv.sh         # Run script (Mac/Linux)
├── run_venv.bat        # Run script (Windows)
├── README.md           # This file
└── src/                # Source code
    ├── gui.py          # User interface
    ├── video_processor.py  # Video processing
    ├── lottie_converter.py # Lottie generation
    └── utils.py        # Utilities
```

## 🎯 Features

- **High Quality Output**: Preserves video quality while compressing
- **Smart Compression**: Advanced algorithms for optimal file size
- **Easy to Use**: Simple GUI and command line interface
- **Cross Platform**: Works on Windows, Mac, and Linux
- **Fast Processing**: Efficient video processing
- **Quality Control**: Adjustable quality and size settings

## 🔧 Troubleshooting

### "Command not found" or "Permission denied"

```bash
# Make script executable (Mac/Linux)
chmod +x run_venv.sh

# Or run directly
python main.py
```

### "Module not found" errors

```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Video won't load

- Make sure video is MP4 format
- Check video isn't corrupted
- Try a different video file

### Output file too large

- Reduce Quality setting (try 70-75%)
- Reduce Max Frames (try 100-120)
- Set smaller Max Size limit

### Poor quality output

- Increase Quality setting (try 85-90%)
- Increase Max Frames (try 180-200)
- Use shorter video clips (10-20 seconds work best)

## 📋 Requirements

- Python 3.8 or newer
- 4GB+ RAM recommended
- MP4 video files
- 100MB+ free disk space

## 🎬 Tips for Best Results

1. **Video Length**: 10-20 seconds work best
2. **Video Quality**: Use high-quality source videos
3. **Resolution**: 720p-1080p recommended
4. **Content**: Simple animations compress better than complex scenes
5. **Testing**: Try different quality settings to find your preferred balance

## 📄 License

This project is open source. Feel free to use and modify as needed.

---

**Need Help?** Check the troubleshooting section above or create an issue in the project repository.
