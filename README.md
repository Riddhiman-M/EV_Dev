# EV_Dev
<h3>This repo presents a method to develop an efficient distribution of Charging Infrastructure for Electric Vehicles.</h3>

First in order to concoct the input we gather vehicle densities from different video feed at different places. <br>
This is done using <b>Computer vision</b>. We use <b>YOLO</b> Detection System (using Deep Neural Network) after training it on COCO Dataset. 
Vehicles are detected from the video feed and classified into Vehicle Types (Cars/ Motorbikes/ Trucks/Buses). The number of vehicle types detected is then stored in a csv file for further processing.<br><br>

The known vehicle densities at various co-ordinates are then processed using <b>K-Means Clustering</b>. Using the User Input of number of clusters to be formed (i.e., number of Charging Stations to be set up), we find the optimum K-Means Cluster centers using Weighted Means in K-Means Clustering.<br><br>

<h3>K-Means Clustering Technique</h3>
<img src="https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning.png" alt="K-Means">
