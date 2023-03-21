# Airline_Delay_Analysis

## Domain Background: 
This project has been inherited from The Airline Domain. In Airline, if you want to travel or anyone who wants to travel, he must book the flight from one place to another. 

There are number of factors, which can impact the flight journey like Weather, flight departure time, boarding gate time and departure time etc. Keeping these factors in mind, we can decide that particular aircraft can be landed or arrive on time or not or how much it will be delay.  

Every airline has their flight history past journey, which can help them in predicting future flight delay. We can implement a machine learning model, which will help us in the prediction of a flight delay. 

The Motivation behind this project is to optimization of network operation, ground staff management and passenger. 

## Group Roles
- Github Master & Database: [Justin](https://github.com/jhohing/Airline_Delay_Analysis/tree/justin_project)
- Machine Learning: [Hanzian](https://github.com/jhohing/Airline_Delay_Analysis/tree/hanzian_project)
- Data Visualization: [Gideon](https://github.com/jhohing/Airline_Delay_Analysis/tree/gideon_project)

## Communication Protocole
| Tools used | Description |
| --- | --- |
| Github |We will use Github to share our work and follow each team member work|
| Slack |we will use Slack to communicate, to review the code and organize team|
| Zoom | We will be use to facilite all face to face meetings, collaborative tasks|

## Problem Statement: 
Flight delays are a common occurrence in the airline industry, causing significant inconvenience and frustration for passengers. Despite efforts to improve airline operations and reduce delays, they continue to impact the travel experience and cost the industry billions of dollars each year. An analysis of flight delays could involve examining various factors that contribute to delays, such as:
- **Weather conditions**: Extreme weather conditions such as storms, heavy rain, snow, and fog can cause flight delays or cancellations.
- **Air traffic congestion**: Busy airports and crowded airspace can cause delays in takeoff and landing.
- **Technical issues**: Technical issues with aircraft or equipment can cause delays, as well as maintenance or repair work.
- **Crew issues**: Delays can occur if there is a shortage of pilots or flight attendants, or if they are delayed or unavailable due to sickness, injury, or other reasons.
- **Security issues**: Security checks and procedures can cause delays, especially during high-security situations.

Analyzing flight delay data can provide insights into patterns and trends, such as the busiest times of year or the most common reasons for delays. This information can help airlines and airport operators to develop strategies for managing delays and improving the overall travel experience for passengers. It can also inform policy decisions related to air travel, such as regulations related to airport capacity or airline operations. 

## Questions to answer
- How does the overall flight volume vary by month? By day of week?

- What percentage of flights in experienced a departure delay in 2015? Among those flights, what was the average delay time, in minutes?

- How does the % of delayed flights vary throughout the year? 

- How many flights were cancelled in 2015? What % of cancellations were due to weather? What % were due to the Airline/Carrier?

- Which airlines seem to be most and least reliable, in terms of on-time departure?

## Data Set
Records for 5,000,000+ commercial airline flights in 2015, compiled for the U.S. DOT Air Travel Consumer Report. Each record represents a single flight, including the airline name, flight number, origin/destination airport and flight distance, as well as scheduled/actual departure and arrival times.
[Airline Flight Delays](https://www.mavenanalytics.io/data-playground?search=ai).
This corpus has 4 csv files. 
-	flights.csv
- airports.csv
-	airline.csv
-	cancellation_codes.csv

For model prediction, we will use flights.csv. But for data insight, we will use other two also.

### Flights.csv
It contains all the past data related to flight schedule from source to destination. Which we will use in our model to test and train. It has 5819079 samples with 31 feature. As we are going to implement it as classification problem. We need to consider a sample per class. 

| Feature Name | Data Type | Description |
| --- | --- | --- |
| YEAR |Integer|Year of travel|
|MONTH |Integer|Month of journey|
|DAY OF WEEK|Integer|Day of the journey|
|DAY |Integer|Day of week for the given journey|
|AIRLINE|String| Namee of Airlines|
|FLIGHT_NUMBER|String|ID of flight|
|TAIL_NUMBER|String|Aircraft Registration Number|
|ORIGIN_AIRPORT|String|Source airport|
|DESTINATION_AIRPORT|String|Destination airport of journey|
|SCHEDULED_DEPARTURE|Float|Schedule departure time of aircraft|
|DEPARTURE_TIME|Float|Actual departure time of aircraft|
|DEPARTURE_DELAY|Float||Actual Delay in departure of aircraft.|
|TAXI_OUT|Float|Time to leave the gate|
|WHEELS_OFF|Float|Wheels take off from runway|
|SCHEDULED_TIME|Float|Schedule take of time of wheels|
|ELAPSED_TIME|Float|
|AIR_TIME|Float|Arrival time at airport |
|DISTANCE|Integer|The distance between ORIGIN_AIRPORT and DESTINATION_AIRPORT|
|WHEELS_ON|Float|Landing time on the runway|
|TAXI_IN|Float|Reached on the gate|
|SCHEDULED_ARRIVAL|Float|Schedule time to reach on gate|
|ARRIVAL_TIME|Float|Actual arrival time |
|**ARRIVAL_DELAY**|**Float**|**Arrival Delay of flight to reach the destination**|
|DIVERTED|Boolean|Flight diverted in between journey to any other airport|
|CANCELLED|Boolean|Flight got cancelled or not|
|CANCELLATION_REASON|String|What was the reason of cancellation of flight? |
|AIR_SYSTEM_DELAY|Boolean|Air system issues|
|SECURITY_DELAY|Boolean|Security issues|
|AIRLINE_DELAY|Boolean|Airline started delay|
|LATE_AIRCRAFT_DELAY|Boolean|Connecting flight delay |
|WEATHER_DELAY|Boolean|Weather issue

### Airlines.csv
This csv contains information related to airline. It has 2 feature with 15 samples.

| Feature Name | Data Type | Description |
| --- | --- | --- |
| IATA_CODE|	string	|IATA Code for airline. It’s  unique identifier of airline|
|Airline| 	string|	Airline Name|

### Airport.csv
This csv contains information related to airport. It has 7 feature with 322 samples.

| Feature Name | Data Type | Description |
| --- | --- | --- |
 IATA_CODE|	String	|IATA code for airport|
|AIRPORT|	String |	Airport name |
|CITY	|String |	Airport city |
|STATE	|String	|Airport belongs to which state.|
|COUNTRY|	Sting	|Country of airport|
|LATITUDE	|Float|	Geographical location of airport|
|LONGITUDE|	Float	|Geographical location of airport|


## Machine Learning Project Design

Every Machine Learning Project have some steps to achieve the goal. Below the steps or action, we need to perform for any ML project. We will follow the same for this project 

### Language and Libraries 
-	Python 3.X 
-	Tensor flow  
-	Scikit-learn 

### Data Collection

For implementing a machine learning model, we need data. In our case, we will collect data in CSV format with `Flights.csv` wich contains all the data  we need for our purpose.


### Feature Engineering 
Feature engineering is the main step in ML model designing. In this, we will do the feature analysis, which feature is more relevant and which are less impacting the outcome. We will use the column **Arrival delays** to cover our feature because our project is based on the analysis of the Airline Flight Delays. 

Dataset in flights.csv is high in volume(~5819079). For computing, it will require high computation power and time. So for this project we are going to use only one month data (July).

### Normalization
In feature Engineering, we will do feature normalization. So because of high magntiude one should not dominate another feature. 
It very important step in the machine learning model. It can drastically impact model performance.

### Train & Test Dataset 
We will Split the dataset into train and test, Training set we will use for our training and testing set for model validation. 

### Model Training & Testing 
By splitting the dataset into a train and test set, we can use the train set to build and train the model and then evaluate its performance on the test set. It is important to note that the train-test split should be done randomly to avoid bias and ensure that the test set is representative of the overall population.

### Model fitting 	
The main goal of fitting a model is to find the best set of parameters or weights that minimize the error between the predicted outputs and the actual outputs of the training data. Once the model has been fit to the training data, it can be used to make predictions.

### Choice of the Model 
Selecting best final model for our purpose. For our project, we will 3 ML models to get the highest accuracy of prediction.

| ML Models | Description | Benefits |
| --- | --- | --- |
| Logistic regression |Logistic regression is a popular statistical model that is commonly used for classification problems. It is a type of supervised learning algorithm that can be used for both binary and multi-class classification tasks.|Logistic regression is a simple and easy-to-understand model. It does not require a lot of computational resources and can be implemented easily. The coefficients in the logistic regression model can be interpreted as the effect of the corresponding input variables on the output variable, making it easy to explain the results to stakeholders.|
| Decision Tree Classifier |Decision tree classifiers are a popular machine learning algorithm used for classification tasks. They are tree-like models that map observations about an item to conclusions about its target value| Decision trees are intuitive and easy to interpret. They can be visualized, making it easy to understand the decision-making process. Decision trees are not sensitive to the scale of the data, and they don't require normalization or scaling of the input features.|
| Random Forest |Random forest is a popular machine learning algorithm that is commonly used for classification, regression, and other tasks. It is an ensemble learning technique that combines multiple decision trees to make more accurate and robust predictions.| Random forests are known for their high accuracy, especially when compared to other single algorithm models. This is because it combines multiple decision trees, which helps to reduce the variance and improve the predictive power. Random forests can handle large datasets with high dimensionality and a large number of features. It is also scalable, making it easy to work with larger datasets. This model is less prone to overfitting and are more robust to outliers and noise in the data.|

### Machine Learning Visualization 
With the help of data visualization, we will try to get insight of data. In visualization, we can see the correlation in between features of data set like Confusion Matrix & ROC Curve. 
-	`Confusion Matrix Plot`: A confusion matrix can help you visualize the performance of your model in terms of true positive, true negative, false positive, and false negative predictions.

- `ROC Curve Plot`: The ROC (Receiver Operating Characteristic) curve can help you evaluate the trade-off between true positive rateand false positive rate of your model at different classification thresholds.

### Prediction Analysis

- Logistic regression model was trained and tested using a train-test split of the data. The model achieved an accuracy score of 0.87, which means that it correctly classified 87% of the observations in the test set. The recall score is 0.89, which means that the model correctly identified 89% of the positive cases in the test set. The precision score is 0.75, which means that when the model predicted a positive case, it was correct 75% of the time.

- Decision tree classifier model achieved an accuracy score of 0.83, which means that it correctly classified 83% of the observations in the test set.

- Random forest classifier model was trained and tested using a train-test split of the data, similar to the previous models. The model achieved an accuracy score of 0.88, which means that it correctly classified 88% of the observations in the test set.
The recall score is 0.89, which means that the model correctly identified 89% of the positive cases in the test set. The precision score is 0.78, which means that when the model predicted a positive case, it was correct 78% of the time.

Compared to the logistic regression and decision tree classifier models, the random forest classifier model performed slightly better in terms of accuracy and recall but performed similarly in terms of precision. Random forest is a more powerful and flexible model than decision tree and logistic regression models, and it can capture more complex relationships among the features in the dataset.

### References
- https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
- https://machinelearningmastery.com/an-introduction-to-feature-selection/
- https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0

## Visualization (Dashbord)

### Summary
![Dashbord](https://github.com/jhohing/Airline_Delay_Analysis/blob/hanzian_project/Dashbord%20Viz/Screenshot%202023-03-16%20at%207.23.07%20PM.png)

- Most delays happens in the month of Febuary,June,January and March 
- Flights are mostly on-time in the month of september and October 
- Weather is primary reason for cancellations 
- ~62% of flights are on time, 36% of flights are delayed; 1% are cancelled.

### Airlines Analysis Summary

![](https://github.com/jhohing/Airline_Delay_Analysis/blob/hanzian_project/Dashbord%20Viz/Screenshot%202023-03-16%20at%207.33.58%20PM.png)

- Spirit Airlines is at the top of delayed airline list followed by Frontier Airlines 
- Delta Airlines is at the top of on-time airline list followed by Alaska Airline, difference between these two is of 4pp (pp- percent piont) 
- Most cancellations is done by American Eagle (~5%) and the least cancellation is done by Hawaiian Airline (~0.2%).

 ### Delay Analysis
 
 ![](https://github.com/jhohing/Airline_Delay_Analysis/blob/hanzian_project/Dashbord%20Viz/Screenshot%202023-03-16%20at%207.34.13%20PM.png)
 
- The major contributors to delay are from Airline followed by Air 
- System Security delay has the least impact on overall delay 
- Southwest Airline is at the top in the delayed airlines list as per flight count 
- Spirit Airline is at the top in the delayed airlines list as per the percent of flight

 ### Conclusion
 
In conclusion, the data provided suggests that flight delays and cancellations are a common occurrence in the airline industry, with weather and airline operational issues being among the main contributing factors. While airlines themselves are often the primary cause of delays, air traffic control systems can also play a significant role.

When looking at specific airlines, it appears that Southwest and Spirit Airlines are among the most frequently delayed carriers, while Delta and Alaska Airlines tend to have the best on-time performance. American Eagle has the highest rate of cancellations, while Hawaiian Airlines has the lowest.

While this information can be helpful for travelers when making decisions about which airline to book with, it's important to remember that delays and cancellations can happen with any carrier. It's always a good idea to plan ahead and be prepared for potential disruptions, such as by booking flexible tickets and having backup travel plans in place.


