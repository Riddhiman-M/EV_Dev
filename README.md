# EV Charging Station Distributor
<h4>This repo presents a method to develop an efficient distribution of Charging Infrastructure for Electric Vehicles.</h4>

### Step 1:
In order to concoct the input we gather vehicle densities from traffic video feed at different places. This is done using <b>Computer vision</b>.<br>
We use the <b>YOLO</b> Detection System (which implements a Deep Neural Network) after training it on [COCO Dataset](https://cocodataset.org/#home). 
Vehicles are detected from the video feed and classified into Vehicle Types (Cars/ Motorbikes/ Trucks/ Buses). The counts of each vehicle type is then stored in a .csv file for further processing.


<h3>Vehicle Detection and Count</h3>

 Raw Input Video frame | Processed Output Video frame 
 :----------------:    |      :------------:         
 <img src="https://user-images.githubusercontent.com/89708853/206383441-dc409faf-0d31-48e6-a7e1-925a37e00e51.png" width=80% height=80%> | <img src="https://user-images.githubusercontent.com/89708853/177213841-9aac4ded-61f7-4549-9fee-059e4a6d091b.png" width=80% height=80%>

<br>

### Step 2:
Processed the data (vehicle co-ordinates) based on statistical metrics to remove outliers from the data (such as high amount of traffic because of a certain event at a paticular place on a single day may cause a skew).

### Step 3:
Group the known vehicle densities at various co-ordinates using <b>K-Means Clustering</b>. Using the User Input of number of clusters to be formed (i.e., number of Charging Stations to be set up), we find the optimum K-Means Cluster centers using Weighted Means in K-Means Clustering.<br><br>

<h3>K-Means Clustering Technique</h3>
<img src="https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning.png" alt="K-Means" width=40% height=40%>
<br>
<h3>Visualized output</h3>
<img src="https://user-images.githubusercontent.com/89708853/206381803-e4e0c2a6-f8e4-4c06-a042-64373376ecd8.png" width=60% height=60%>
