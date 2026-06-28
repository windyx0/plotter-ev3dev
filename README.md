[🇷🇺 Русская версия](README_RU.md)

# Plott3r 🤖🖍️

Plott3r is a hardware-software complex for creating drawings using LEGO Mindstorms EV3. You upload any image via a convenient web interface, the computer algorithms process it (find contours, optimize paths), generate G-code, and send it directly to the EV3 robot via Wi-Fi for plotting!

## 🌟 Key Features
- **Web Interface** — a beautiful and modern UI for uploading and processing images right in your browser.
- **Cross-Platform Support** — the server can run on Windows, Linux, macOS, and even an Android phone (via Termux).
- **Multiple Modes** — automatic conversion of photos to lines (via Canny filters) or completely manual drawing of custom contours.
- **Computer Vision & Optimization** — utilizes OpenCV and Ramer-Douglas-Peucker (RDP) algorithms to achieve the perfect balance between detail and print speed.
- **Calibration & Control** — move the carriage, change arc aggressiveness, and park the robot directly from the EV3 or your PC.

---

## 🛠️ Server Setup (PC / Smartphone)

All heavy image processing happens on the server. For convenience, we have prepared installation scripts for various systems.

**1. Clone the repository or download the project files.**
Open the terminal in the project folder (`plott3r_project`).

**2. Run the installation script for your OS:**

- **For Windows:**
  ```bat
  install\install_windows.bat
  ```
- **For Linux (Ubuntu, Debian, etc.):**
  ```bash
  bash install/install_linux.sh
  ```
- **For macOS:**
  ```bash
  bash install/install_macos.sh
  ```
- **For Android (Termux):**
  ```bash
  bash install/install_termux.sh
  ```
  *(The script will automatically fetch heavy packages like OpenCV and NumPy from specialized repositories to avoid compiling them on your phone).*

---

## 🚀 Getting Started

### Step 1. Start the Server
After installation, you can launch the web interface using one of the startup scripts:
- On Windows: `start_windows.bat`
- On Linux: `bash start_linux.sh`
- On macOS: `bash start_macos.sh`
- On Termux: `bash start_termux.sh`

Once started, the terminal will display an address like: `http://192.168.1.xxx:5000`. Open this link in your web browser.

### Step 2. Prepare the EV3
1. Connect your EV3 to the same Wi-Fi network as your PC/Smartphone.
2. Copy the `ev3_side` folder to your EV3 block (e.g., via VS Code with the ev3dev extension).
3. On the EV3, run the `main_menu.py` script.
4. In the menu, select `Print From Server` (the robot will display its IP address on the screen).

### Step 3. Print!
1. In the web interface, select an image and adjust the sliders to your liking.
2. Click **"Process"** — a preview of the result will appear on the right.
3. Go to the **"Send to EV3"** tab.
4. Enter the IP address shown on the robot's screen and click **"Send"**.
5. Watch Plott3r do its magic! 🎨

---

## 🔧 Project Structure
- `computer_side/` — Python backend (Flask) and Frontend source code (HTML/JS/CSS).
- `ev3_side/` — code that runs directly on the EV3 (motor control, G-code parsing).
- `install/` — scripts for automatic dependency installation.
- `start_*.sh` — scripts for a quick web server startup.

## ⚙️ EV3 Menu
- **Continue Unleashed:** Resume printing if it was interrupted or canceled (progress is saved).
- **Print From Server:** Wait for G-code commands over Wi-Fi.
- **Select File:** Print G-code from a local file on the EV3 itself.
- **Calibrate Menu:** 
  - `Full Calibrate` — calibrate X and park the paper.
  - `Eject Paper` — eject the paper.
  - `Calibrate X` — calibrate the pen carriage.
  - `Pen Test` — raise/lower the pen (Enter), exit (Left/Right).

> 💡 **Tip:** During printing, you can press the `Left`, `Right`, or `Back` buttons on the EV3 at any time to cancel the print. Your progress will be saved automatically!
