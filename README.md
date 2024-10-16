# Sea_Trails

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
| **Date**           | Represents the year and month when the sea trial was conducted.                                      |
| **지연 여부**        | Indicates whether there was a delay during the trial (normal operation / delayed operation).        |
| **업체 타입**        | The type of trial operator (direct operation / cooperative operation).                              |
| **선종**            | The specific type of ship.                                                                           |
| **시운전 일수**      | The number of days of the sea trial.                                                                |
| **HFO, MFO**       | The amount of fuel used by the ship.                                                                 |
| **선장_투입~노무_안전_투입** | Represents the personnel involved in the sea trial.                                          |

##Analyze

