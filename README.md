
# The Official Code of VariGAN: Enhancing Image Style Transfer by UNet Generator & Auto-encoding Discriminator via GAN

### Environmental Requirement
- Windows
- Python 3.X
- CPU or NVIDIA GPU + CUDA CuDNN
- You can refer to `requirements.txt` for specific versions used in this project


![image](https://github.com/Gdw040199/VariGAN/blob/main/image/1.png)
![image](https://github.com/Gdw040199/VariGAN/blob/main/image/2.jpg)
# Configuration Procedure

## 1. Clone Project
1. Open PyCharm, Tools -> Space -> Clone Repository
2. Repository URL: Enter the GitHub project URL

## 2. Install Necessary Packages
Create an environment using Anaconda and pip to install the necessary libraries:
1. Open the Anaconda Prompt
2. Create a virtual environment: `conda create -n pytorch_VariGAN python=3.8`
3. Enter the virtual environment: `conda activate pytorch_VariGAN`
4. Activate the environment on the project directory and install the required dependencies: `pip install -r requirements.txt`

## 3. Download Datasets
1. The dataset for this project is identical to *CycleGAN*. Download the dataset from the link below:
2. Extract the official dataset into the project's `datasets` directory: [Datasets](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)

## 4. Train the Backbone
1. Set the debugging configuration before running `train.py`.
2. Enter parameters in the file configuration interface in the format: `--dataroot [path to your training set tastA] `
3. Modify training parameters in `train.py` as needed, such as training times, batch size, etc.
4. After starting the service, run the `train.py` file to start training the model.
5. In `Datasets.py`, you should change the root address of the `ImageData` function to the data set you want to train.

## 5. Custom Training Using Your Own Training Set
1. Naming:
   - The training sets are named `trainA` and `trainB`
   - The test sets are named `tastA` and `tastB`
2. Train directly, the same way as the previous step.

## 6. Using VariGAN Weight File After Training
1. Find the path to the `testA` or `testB` image that you want to convert.
2. Rename the weight file you want to use to `G_AB_4.pth` and `G_BA_4.pth`.
3. Before running `test.py`, before that, you should change the root of the weight file.
4. Run the `test.py` file for testing, and the generated converted image will be saved in the `results` directory.

## 7. Evaluation
1. The evaluation code is in the `evaluation` directory.
2. For FCN evaluation, you can use the `Train_FCN.py` file to train a fcn model. Then, you can use the `evaluation_FCN_scores.py` file to test the model.
3. Before you Do the training FCN, you should label the data set you want to train. The guidelines are this [link](https://github.com/wkentaro/labelme). 
4. Then, you should change the root of the data set (ground truth) in the `Train_FCN.py` file. Then, you can train the FCN model.
5. After training the FCN model, you can use the `evaluation_FCN_scores.py` file to test the model.
6. The dataroot in the `evaluation_FCN_scores.py` file should be the root of the data set you want to test. 
4. For other evaluation methods like FID, PSNR, SSIM, you can use the `score_others.py` file to test the generated images.
