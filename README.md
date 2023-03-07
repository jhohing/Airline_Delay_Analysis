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

## Problem Statement: 
Flight delays are a common occurrence in the airline industry, causing significant inconvenience and frustration for passengers. Despite efforts to improve airline operations and reduce delays, they continue to impact the travel experience and cost the industry billions of dollars each year. An analysis of flight delays could involve examining various factors that contribute to delays, such as:
- **Weather conditions**: Extreme weather conditions such as storms, heavy rain, snow, and fog can cause flight delays or cancellations.
- **Air traffic congestion**: Busy airports and crowded airspace can cause delays in takeoff and landing.
- **Technical issues**: Technical issues with aircraft or equipment can cause delays, as well as maintenance or repair work.
- **Crew issues**: Delays can occur if there is a shortage of pilots or flight attendants, or if they are delayed or unavailable due to sickness, injury, or other reasons.
- **Security issues**: Security checks and procedures can cause delays, especially during high-security situations.

Analyzing flight delay data can provide insights into patterns and trends, such as the busiest times of year or the most common reasons for delays. This information can help airlines and airport operators to develop strategies for managing delays and improving the overall travel experience for passengers. It can also inform policy decisions related to air travel, such as regulations related to airport capacity or airline operations. 

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
|**ARRIVAL_DELAY|Float|Arrival Delay of flight to reach the destination**|
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




