# PyGenStabilityOne

PygenStabilityOne is a community detection algorithm that leverages Markov stability (through the [PyGenStability](https://github.com/barahona-research-group/PyGenStability) algorithm) as well as machine learning to return a single **robust** partition at a **suitable scale** that is consistent with the structure of the network, without requiring any information/assumption about the partition (number of communities, scale, resolution) from the user.

![flowchart-pygenone](https://github.com/user-attachments/assets/38925984-4076-4a6d-949f-fab86add33db)

## Installation
First, clone the repository with the following command:
```
git clone https://github.com/saref/PyGenStabilityOne.git
```
Then, either manually install all of the dependencies listed in [requirements.txt](requirements.txt), or install them all at once with:
```
pip install -r requirements.txt
```
_Note: It is recommended to use this library with `Python 3.11` in order to avoid dependency issues._

## Usage
To use PyGenStabilityOne, simply:
- Import the `pygenstability_one` function from the `pygenstability_one` library.
- Call `pygenstability_one` with an `nx.Graph` network as input.

Then, PygenStabilityOne will return a list of the communities detected in the network (where each community is represented by a list of node labels).

```py
from pygenstability_one import pygenstability_one
graph = nx.read_gml("ABCD_0.gml", destringizer=int) # The graph should be an nx.Graph graph
coms = pygenstability_one(graph) # The returned value is a list of detected communities
```
See [example.ipynb](example.ipynb) for example code that can be run locally.
