# Workshop In Autonomous Systems Final Project Roei Levi

## 🚀 Installation

1. **Install CARLA 0.9.15**  
   Download and install the CARLA 0.9.15 simulator from the [official website](https://carla.org/).

2. **Create a Python 3.8.10 virtual environment and activate it.**  

3. **Install PyTorch separately**
   Install the specific CUDA‑enabled versions of Torch, Torchvision, and Torchaudio (these cannot be installed via the `requirements.txt`):

   ```bash
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install remaining Python dependencies**
   With the virtual environment still active, run:

   ```bash
   pip install -r requirements.txt
   ```

5. **Clone Deep SORT**
   Clone the [Deep SORT repository](https://github.com/nwojke/deep_sort):

   ```bash
   git clone https://github.com/nwojke/deep_sort
   ```

   Rename the cloned folder to `deep_sort` and place it inside the **project directory**.

6. **Download and place model weights**
   Download the required weights from [this Google Drive folder](https://drive.google.com/drive/folders/1PH9kSqqnMTOlaNroHKJqVCkpnCFZIXix?usp=sharing):

   * Move `yolo12n.pt` to:

     ```
     perception/object_detection/yolo12n.pt
     ```
   * Move `mars-small128.pb` to:

     ```
     perception/object_tracking/mars-small128.pb
     ```

---

## ▶️ Running the System

1. **Set the project directory**
   Ensure your **current working directory** in the terminal is the project root (where `main.py` is located).

2. **Required CLI arguments**

   * `--carla-api-path`: **Required.** Set this to the absolute path of your CARLA `PythonAPI/carla` folder.
     Example on Windows:

     ```bash
     --carla-api-path D:\Carla\PythonAPI
     ```
   * `--yolo-weights-path` and `--deepsort-enc-weights-path` default to:

     ```
     perception/object_detection/yolo12n.pt
     perception/object_tracking/mars-small128.pb
     ```

     Make sure the files are actually present in these locations or adjust the arguments accordingly.

3. **Run the system**

   * First, launch the CARLA simulator.
   * After CARLA has fully loaded, run:

     ```bash
     python main.py --carla-api-path CARLA_API_PATH
     ```

     Replace `CARLA_API_PATH` with your absolute CARLA API path.

4. **Explore more options**
   To see all available arguments and their descriptions:

   ```bash
   python main.py --help
   ```