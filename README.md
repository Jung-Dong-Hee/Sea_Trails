# Sea Trials Cost Analyze

## Sea Trails
A sea trial is the testing phase of a watercraft (including boats, ships, and submarines). It is also referred to as a "shakedown cruise" by many naval personnel. It is usually the last phase of construction and takes place on open water, and it can last from a few hours to many days.

Sea trials are conducted to measure a vessel's performance and general seaworthiness. Testing of a vessel's speed, maneuverability, equipment and safety features are usually conducted. Usually in attendance are technical representatives from the builder (and from builders of major systems), governing and certification officials, and representatives of the owners. Successful sea trials subsequently lead to a vessel's certification for commissioning and acceptance by its owner.

Although sea trials are commonly thought to be conducted only on new-built vessels (referred by shipbuilders as 'builders trials'), they are regularly conducted on commissioned vessels as well. In new vessels, they are used to determine conformance to construction specifications. On commissioned vessels, they are generally used to confirm the impact of any modifications.

Sea trials can also refer to a short test trip undertaken by a prospective buyer of a new or used vessel as one determining factor in whether to purchase the vessel.

https://en.wikipedia.org/wiki/Sea_trial

## Data
Collecting data related to sea trials is challenging. Due to the lack of data on trials, we generated the dataset. In the case of HFO-MFO and 선장_투입-노무_안전_투입, the personnel input numbers were generated through our own simulations.
We generated data for ship sea trials from 2015 to July 2024 and analyzed the data using sampling methods.

The following data was generated considering inflation rates in South Korea:

교통비, 도선비, 임시항해검사비, 자차수정비, 양식, 한식, 용도품_침구, 용도품_물품, 예선료, 통선비

The following data was generated considering wage increase rates in South Korea:

선장비용, 타수비용, 노무원비용

Data was generated with careful consideration to reflect realistic scenarios and conditions for sea trials. The aim was to maintain data consistency while accommodating variations observed due to economic factors such as inflation and wage changes. The primary goal was to analyze sea trial costs based on ship types. Additionally, we plan to add future cost estimation and root cause analysis.

## Data Description
| Feature            | Description                                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------|
| **Date**           | The date of the trial operation.                                                                     |
| **지연 여부**      | Whether a delay occurred during the trial operation. (Normal operation / Delayed operation)          |
| **업체 타입**      | The type of company conducting the trial operation. (Direct = 직영, Cooperation = 협력)               |
| **선종**           | The type of vessel on which the trial operation was conducted.                                       |
| **시운전 일수**    | The actual duration of the trial operation. (Unit: Days)                                             |
| **HFO**            | The amount of HFO used during the trial operation. (Unit: ℓ)                                         |
| **MFO**            | The amount of MFO used during the trial operation. (Unit: ℓ)                                         |
| **선장_투입~노무_안전_투입** | The number of personnel involved in the trial operation. (Unit: Persons)                   |

#### Question : What is "Other cost" and what data columns does it contain?
#### Answer : Refers to additional expenses incurred during the sea trials. It includes all costs except for "Labor Cost" and "Fuel Cost."

#### Column : "용도품 침구 및 물품", "예선료", "통선비", "양식", "한식"

'용도품 침구 및 물품' : Refers to supplies and bedding items necessary for the navigators during sea trials.

'예선료(Pilotage Fee)' : Pilotage fees are costs associated with hiring a licensed pilot who navigates the vessel through specific waterways, ports, or narrow passages. These fees cover the expertise needed to ensure safe and efficient maneuvering, especially in complex or congested areas, during sea trials and other navigational activities.

'통선비(Tugboat Fee)' : Tugboat fees are expenses incurred for using tugboats to assist the main vessel with docking, undocking, or navigating in confined spaces. Tugboats provide essential support, particularly when the ship lacks sufficient maneuverability or needs additional control during certain parts of the voyage or testing phase.

'양식' : Western-style food consumed by the ship's owner during sea trials.
'한식' : Korean-style food consumed by the crew members during sea trials.

# Analyze

## 1. Number of sea trials tests per month
Analyzing the number of monthly test runs for normal operation and delayed operation
<figure>
  <p align="center">
    <img src="Fig/Number of sea trials tests per month.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

## 2. Cost analysis by category
Analysis of fuel costs, labor costs, other costs, and total costs for normal operation, delayed operation, and total operation of the ship type.
<figure>
  <p align="center">
    <img src="Fig/Cost analysis by category.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

## 3. Cost analysis by normal/delayed items
Analysis of total costs, labor costs, and fuel costs for normal, delayed, and total operation of a ship type over time.

### 1.Total cost
<figure>
  <p align="center">
    <img src="Fig/Total Cost TS.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

### 2.Labor cost
<figure>
  <p align="center">
    <img src="Fig/Labor Cost Ts.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

### 3.Fuel cost
<figure>
  <p align="center">
    <img src="Fig/Fuel Cost Ts.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

## 4. Cost Forecast
We forecast the total expenses for each ship type by predicting the individual cost items. The forecast is displayed in both graph and table formats. To view specific forecast values, you can click on the table.

### 1. Labor cost

<figure>
  <p align="center">
    <img src="Fig/Labor cost prediction.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

### 2. Other cost

<figure>
  <p align="center">
    <img src="Fig/Other cost prediction.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

### 3. Fuel cost

<figure>
  <p align="center">
    <img src="Fig/Fuel cost prediction.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

### For the Fuel Cost Forecast section, the process is divided into two parts:

Oil Price: Based on international commodity prices, the oil price forecast is derived from HFO and MFO price data. This forecast predicts future oil prices independently of year and ship type.

Exchange Rate: The exchange rate forecast uses USD-KRW and EUR-KRW exchange rates from the global financial markets. This prediction estimates future exchange rates independently of year and ship type.

### Part 1 - HFO & MFO cost prediction

<figure>
  <p align="center">
    <img src="Fig/HFO MFO cost prediction.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

### Part 2 - Dollar & Euro prediciton

<figure>
  <p align="center">
    <img src="Fig/HFO MFO cost prediction.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

## 5. Cause Analysis Report
Based on the forecasted costs of sea trials, we analyze the predicted costs for 2025 compared to 2024. Each item represents necessary components for trial operations.

"Labor cost," "Other cost," "Fuel cost"

The ship type can be selected, and the desired month in 2025 can be chosen for viewing. Costs are displayed in a table format, showing the average costs for each item based on 2024 and the predicted costs for 2025 beside them. The rate of increase from the previous year is also shown. Finally, the selected ship type, item, and increase rate are presented as results.

EX) For the "2500TEU_EGCS_1st" ship type, "Other Cost" showed the largest variation, with an increase of 4.90% of the total expenses.

<figure>
  <p align="center">
    <img src="Fig/Cause Analysis Report.PNG" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

## 6. Radar Chart
Analysis by comparing the ratio between normal and delayed operation of fuel costs, sea trials navigator costs, ship maintenance & management team costs, total costs, and other costs.

<figure>
  <p align="center">
    <img src="Fig/Radar Chart.png" alt="Trulli" style="width:80%">
    <figcaption align = "center"></figcaption>
  </p>
</figure>

# Cite
https://sea-trails.streamlit.app/
