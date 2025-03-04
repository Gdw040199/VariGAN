# VariGAN: Enhancing Image Style Transfer with UNet Generator & Auto-encoding Discriminator via GAN

### System Requirements
- **Operating System**: Linux and Windows
- **Python Version**: Python 3.8
- **Hardware**: NVIDIA GPU with CUDA and CuDNN support
- **Dependencies**: Refer to `requirements.txt` for the specific versions of required packages.

---

## Setup Instructions

### 1. Clone the Repository
1. Open **PyCharm**.
2. Navigate to **Tools -> Space -> Clone Repository**.
3. Enter the GitHub project URL in the **Repository URL** field and clone the project.

---

### 2. Install Required Packages
Use **Anaconda** to create a virtual environment and install the necessary dependencies:
1. Open the **Anaconda Prompt**.
2. Create a virtual environment:  
   ```bash
   conda create -n pytorch_VariGAN python=3.8
   ```
3. Activate the virtual environment:  
   ```bash
   conda activate pytorch_VariGAN
   ```
4. Navigate to the project directory and install the required libraries:  
   ```bash
   pip install -r requirements.txt
   ```

---

### 3. Download the Dataset
1. This project uses the same dataset as **CycleGAN**. Download the dataset from the following link:  
   [CycleGAN Datasets](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)
2. Extract the dataset into the `datasets` directory within the project.

---

### 4. Train the Model
1. Configure the debugging settings before running `train.py`.
2. Set the training parameters in the file configuration interface using the following format:  
   ```bash
   --dataroot [path to your training dataset tastA]
   ```
3. Modify training parameters (e.g., number of epochs, batch size) in `train.py` as needed.
4. Start the service and run the `train.py` script to begin training the model.
5. In `Datasets.py`, update the `ImageData` function's root path to point to the dataset you want to use.

---

### 5. Custom Training with Your Own Dataset
1. **Dataset Naming Convention**:
   - Training sets: `trainA` and `trainB`
   - Test sets: `tastA` and `tastB`
2. Follow the same training procedure as described in the previous step.

---

### 6. Testing with Trained Weights
1. Locate the `testA` or `testB` images that you want to convert.
2. Rename the weight files you want to use to `G_AB_4.pth` and `G_BA_4.pth`.
3. Update the weight file root path in the `test.py` script.
4. Run the `test.py` script to generate converted images. The output images will be saved in the `results` directory.

---

### 7. Evaluation
#### 7.1 FCN Evaluation
1. Navigate to the `evaluation` directory and open the `label` folder. Follow these steps:
   - **Step 1**: Prepare the Train datasets and Test Datasets. Edit the `dataset` folder under `label`. Place the images in the `imgs` folder and specify the classes you want to segment in the images. For the folder `test`, you should also make the same folder named `dataset` and put the test imgs in it.
   - **Step 2**: Use the `labelme` library to annotate the images. Navigate to the `dataset` directory and run the following command:  
     ```bash
     labelme imgs --output jsons --nodata --autosave --labels labels.txt
     ```
   - **Step 3**: Convert the `jsons` dataset into the VOC format by running the `img2voc.py` script:  
     ```bash
     python img2voc.py
     ```
     This will create a `dataset_voc` folder containing the following subdirectories:
     - `JPEGImages`: Stores the original images.
     - `SegmentationClass`: Stores the segmentation masks in PNG format.
     - `SegmentationClassnpy`: Stores the segmentation masks in NumPy format.
     - `SegmentationClassVisualization`: Stores visualized segmentation masks overlaid on the original images.
     - `class_names.txt`: Contains the category names, including the background.
   - **Step 4**: Navigate to the `pytorch_segmentation` folder to train the FCN network. Run the following commands:
     ```bash
     cd pytorch_segmentation
     python train.py --config config.json
     ```
     Use the `txt_build.py` script to create label files for dividing the dataset into training and validation subsets.
   - **Step 5**: To test the trained FCN network, place the `best_model.pth` file in the `ckpt` folder under the `test` directory. Then, run the following command:
     ```bash
     python main.py
     ```

#### 7.2 Other Evaluation Methods
1. To calculate model parameters, use the `test_para.py` script located in the `evaluation` directory.

---

### Notes
- Ensure proper dataset organization and file paths during training and testing.
- For additional details or troubleshooting, refer to the comments in the provided scripts.

--- 
