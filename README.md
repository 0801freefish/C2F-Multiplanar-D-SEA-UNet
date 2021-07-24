## Coarse-to-fine Multiplanar D-SEA UNet for Automatic 3D Carotid Segmentation in CTA Images


### 1. Install the virtual environment annconda
Installation tutorial: https://blog.csdn.net/yaohuan2017/article/details/108669791
Configuring environment variables tutorial: https://blog.csdn.net/weixin_38705903/article/details/86533863
Add Tsinghua source tutorial: https://blog.csdn.net/qq_45688354/article/details/108014189
Create a virtual environment 
    ```conda create -n multiplanarunet python=3.6```
Activate the virtual environment: 
    ```source activate multiplanarunet```
Install tensorflow-gpu:
    ```conda install tensorflow-gpu=1.11.0```

### 2. Install mulplanar unet (already configured, no need to repeat configuration)
The github link of the project: https://github.com/perslev/MultiPlanarUNet
(1) Open the terminal and run in the /media/bjtu/new volume/yyy/MultiPlanarUNet-master folder: sudo /home/bjtu/anaconda3/envs/multiplanarunet/bin/python setup.py install
What is missing in the process and what to download: sudo /home/bjtu/anaconda3/envs/multiplanarunet/bin/pip install installation package name
(2) Change the network model unet to D-SEA UNet: Open the folder /home/bjtu/anaconda3/envs/multiplanarunet/lib/python3.6/site-packages/MultiPlanarUNet-0.2.3-py3.6.egg/MultiPlanarUNet /models
Name the original unet.py unet-initial.py, and name the modified D-SEA UNet network unet.py

### 3. Coarse segmentation
(1) Modify the network model (remove the attention mechanism)
Open the folder /home/bjtu/anaconda3/envs/multiplanarunet/lib/python3.6/site-packages/MultiPlanarUNet-0.2.3-py3.6.egg/MultiPlanarUNet/models
Add the # comment at the beginning of line 193 in the unet.py file
(2) Data set preparation: downsample 512*512*512 large images (training set: New_add, test set: New_add_test) to 144*144*144 (New_add_resize, New_add_test_resize)
Open the terminal under the folder /media/bjtu/new volume/yyy/data/preprocessed (right click-open in terminal):
Change the 30th line in resize.py to path ='./New_add' and run python resize1.py
Change line 31 in resize.py to size=[144,144,144]
Change the 30th line in resize.py to path ='./New_add_test' and run python resize1.py
(3) Convert the data set to the format required by multiplanar unet
Open the terminal under the folder /media/bjtu/new volume/yyy/MultiPlanarUNet-master:
Modify lines 20-22 of preprocess.py: trainpath="New_add_resize"; testpath="New_add_test_resize";target_folder="./data_folder-pred1"
Run python preprocess.py
The converted data set folder is ./data_folder-pred1
(4) Initialize the project
Open the terminal under the folder /media/bjtu/new volume/yyy/MultiPlanarUNet-master:
source activate multiplanarunet #Activate the virtual environment
mp init_project --name labelsegpred1 --data_dir ./data_folder-pred1 #Create project folder labelsegpred1 and configuration file train_hparams.yaml
cd labelsegpred1 #Enter the project folder
Modify the parameter depth: 4 in the configuration file train_hparams.yaml; loss: "sparse_categorical_crossentropy"; batch_size: 8;n_epochs: 150; optimizer_kwargs: {lr: 5.0e-04, decay: 0.0, beta_1: 0.9, beta_2: 0.999, epsilon : 1.0e-8};
(5) Start training
Continue to run in the terminal:
mp train --no_val --overwrite
Run after training:
mp train_fusion
(6) Start the test (the trained model is already available, you can directly open the terminal under the folder /media/bjtu/new volume/yyy/MultiPlanarUNet-master/labelsegpred1, run source activate multiplanarunet and continue the following test operations )
Continue to run in the terminal:
mp predict --overwrite
Copy the generated test results./predictions/nii_files to the /media/bjtu/new volume/yyy folder
(7) Evaluate the test results
Open the terminal in the /media/bjtu/new volume/yyy folder:
Modify line 82 of pro-mul.py: path ='/media/bjtu/new volume/yyy/data/preprocessed/New_add_test_resize'
Run python pro-mul.py
Get the prediction folder (test set CTA image: data_data_t1.nii.gz, test set prediction result: prediction.nii.gz, test set doctor's gold standard: truth.nii.gz)
Run python evaluate.py to output the accuracy of the test set segmentation dice (for example, the coarse segmentation result is 0.8974)
For ease of management, rename the prediction folder to prediction-0.8974
Delete the ./nii_files folder

### 4. Perform full cropping on the coarse segmentation test results
Open the terminal under the /media/bjtu/new volume/yyy/data/preprocessed folder:
Modify the line 103 of vessel_pred.py predpath ='/media/bjtu/new volume/yyy/prediction-0.8974/'+folder_name+'/prediction.nii.gz'
Run python vessel_pred.py
Get the full cropped folder New-fullcrop-test-pred of the coarse segmentation test results, this folder will be used as the fine segmentation test set to continue the fine segmentation

### 5. Fine segmentation
(1) Modify the network model (plus attention mechanism)
Open the folder /home/bjtu/anaconda3/envs/multiplanarunet/lib/python3.6/site-packages/MultiPlanarUNet-0.2.3-py3.6.egg/MultiPlanarUNet/models
Remove the # comment on line 193 in the unet.py file
(2) Data set preparation: The training set for fine segmentation tailored according to the gold standard: New-fullcrop, and the test set after the coarse segmentation test results are fully trimmed: New-fullcrop-test-pred unified to 160*80*224 (New- fullcrop_resize, New-fullcrop-test-pred_resize)
Open the terminal under the folder /media/bjtu/new volume/yyy/data/preprocessed:
Change the 30th line in resize.py to path ='./New-fullcrop', and run python resize1.py
Change line 31 in resize.py to size=[160,80,224]
Change the 30th line in resize.py to path ='./New-fullcrop-test-pred' and run python resize1.py
(3) Convert the data set to the format required by multiplanar unet
Open the terminal under the folder /media/bjtu/new volume/yyy/MultiPlanarUNet-master:
Modify lines 20-22 of preprocess.py: trainpath="New-fullcrop_resize"; testpath="New-fullcrop-test-pred_resize"; target_folder="./data_folder-pred2-aug"
Run python preprocess.py
The converted data set folder is ./data_folder-pred2-aug
(4) Initialize the project
Open the terminal under the folder /media/bjtu/new volume/yyy/MultiPlanarUNet-master:
source activate multiplanarunet #Activate the virtual environment
mp init_project --name labelsegpred2-aug --data_dir ./data_folder-pred2-aug #Create project folder labelsegpred2-aug and configuration file train_hparams.yaml
cd labelsegpred2-aug #Enter the project folder
Modify the parameter depth: 4 in the configuration file train_hparams.yaml; loss: "sparse_categorical_crossentropy"; batch_size: 8;n_epochs: 150; optimizer_kwargs: {lr: 5.0e-04, decay: 0.0, beta_1: 0.9, beta_2: 0.999, epsilon : 1.0e-8};
(5) Start training
Continue to run in the terminal:
mp train --no_val --overwrite
Run after training:
mp train_fusion
(6) Start the test (the trained model already exists, you can directly open the terminal under the folder /media/bjtu/new volume/yyy/MultiPlanarUNet-master/labelsegpred2-aug, and continue the following after running source activate multiplanarunet Test operation)
Continue to run in the terminal:
mp predict --overwrite
Copy the generated test results./predictions/nii_files to the /media/bjtu/new volume/yyy folder
(7) Evaluate the test results
Open the terminal in the /media/bjtu/new volume/yyy folder:
Modify line 13 of pro-mul.py: path ='/media/bjtu/new volume/yyy/data/preprocessed/New-fullcrop-test-pred_resize'
Run python pro-mul.py
Get the prediction folder (test set CTA image: data_data_t1.nii.gz, test set prediction result: prediction.nii.gz, test set doctor's gold standard: truth.nii.gz)
Run python evaluate.py to output the test set segmentation dice accuracy (for example, the fine segmentation result is 0.9252)
For easier management, rename the prediction folder to pred
iction-0.9252
Delete the ./nii_files folder

### 6. Complete the fine segmentation test results and restore the area without carotid artery that was cropped around during cropping to the original image size of 512*512*512 to obtain the final segmentation result
(1) Open the terminal under the /media/bjtu/new volume/yyy/data/preprocessed folder:
Modify restore.py lines 105-106 predpath1 ='/media/bjtu/new volume/yyy/prediction-0.8974/'+folder_name+'/prediction.nii.gz'#Rough segmentation result;predpath2 ='/media/bjtu /New volume/yyy/prediction-0.9252/'+folder_name+'/prediction.nii.gz'#Fine segmentation result
Run python restore.py
The final segmentation result is located in /media/bjtu/new volume/yyy/prediction
(2) Evaluate the final segmentation result
Open the terminal in the /media/bjtu/new volume/yyy folder:
Run python evaluate.py to output the accuracy of the test set segmentation dice (for example, the final segmentation result is 0.9151)
For ease of management, rename the prediction folder to prediction-0.9151
