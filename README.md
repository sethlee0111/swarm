# Swarm 
Simulation for opportunistic decentralized learning
🐜🐜🐜🐜🐜🐜🐜🐜🐜
## Dependencies
This repo is tested with python3. Packages such as tensorflow, numpy and matplotlib is required to run this package. To bulk install the dependencies,
```
pip3 install -r requirements.txt
```

## Run
### Simulation
Run the simulation by running driver.py and result file will be generated under /hist.
```
python3 driver.py [args]
```
#### Example
For example, you can run simulation with 20 clients for 5 steps with LeviFlightSwarm model and a comparative central model.
```
python3 driver.py --num 20 --steps 5 --swarm 'LeviFightSwarm' --comp_central
```

| Argument     | Description                 | Default     |
| -----------  | --------------------------- | ----------- |
| --num        | set number of clients       | 30          |
| --steps      | set number of steps         | 10          |
| --maxenc     | maximum encounters in step (only in RandomSwarm)  | 5           |
| --avg_epochs | average epochs in step      | 5           |
| --thres      | min. thres. of # of epochs from the common ancestor for exchange    | 3           |
| --stop_enc   | total number of encounters as a stop condition for the entire simulation | None |
| --uni        | when clients encounter, one client takes the others model. They don't share it. However, there can be random occurances of two clients effectively sharing model in a step.    | N/A           |
| --swarm      | type of a swarm             | RandomSwarm |
| --model      | deep learning model         | 2NN         |
| --comp_central | run a centralized model (standard federated model) after the simulation for comparison. Refer to below for the supported options  | 'acc'           |
| --noniid | non-iid mode. set variables in noniid-config.txt   | N/A           |
* `--model` option is not available now and will be set to default (2NN)

### Visualize the result
Run `visualize.py` to visualize the change of accuracy over time

```
python3 visualize.py hist/[history_filename.pickle]
```
| Argument     | Description                 |
| -----------  | --------------------------- |
| --avg_acc    | plot the average accuracy of clients |
| --animate    | animate the accuracies of all clients over the steps |

### More about `--comp_central`
`acc`: centralized model runs until it hits the final average accuracy of the swarm. 
`comm`: centralized model runs so that the most similar number of communications would happen. "Communication" is a client-client or client-server connection being established for exchanging/sending/receiving models. The graph will be displayed over the graph of the decentralized model in 'avg_acc' mode

## Swarm Models
Clients go through steps, and in each step they run training for random epochs, encountering and exchanging models with the other clients.

### RandomSwarm
The clients encounter other clients in a completely random fashion.

### LevyFlightSwarm
The clients have x and y coordinate which follows [Levy Flight](https://en.wikipedia.org/wiki/L%C3%A9vy_flight) random walk. Degree of a direction follows uniform distribution.

## Configure non-iidness
Edit noniid_config.json file to configure the data distribution of clients. You can add a custom non-iid setting in two ways. Two ways of configuration are supported; `auto` and `manual`. In any case, two clients will not have overlapping data.

### 'Auto' configuration (Not supported currently)
| Argument              | Description                          |
| --------------------  | ------------------------------------ |
| name                  | the name of the configuration. Use the name specified here when running a simulation with driver.py  |
| num_lables_per_client | For example, when this is set to 2, the labels per client would be (0,1), (2,3), ... However, the combination is completely random  |
| are_label_sets_unique | if this is set to true, the combinations of the labels per client will be unique to each other. e.g. (0,1),(2,3)|
| noise                 | Different labels that are mixed in the label set |

```
"auto": [
            {
                "name": "my_config",
                "num_lables_per_client": 2,
                "are_label_sets_unique": true,
                "noise": 0.1
            }
        ],
```
### 'Manual' configuration
| Argument              | Description                          |
| --------------------  | ------------------------------------ |
| name                  | the name of the configuration. Use the name specified here when running a simulation with driver.py  |
| label_sets            | the collection of the definition of labels distributed to each client. labels are integers bigger than or equal to 0. |
| label_sets - ratio    | E.g, if ratio is set to 0.2, 20% of the clients ends up with the corresponding label set. For the clients with unspecified ratio, the label sets are distributed evenly. |
| noise                 | The noise labels are never the same labels as the specified ones |

```
"manual": [
            {
                "name": "mnist_config",
                "label_sets": [
                    {
                        "ratio": 0.4,
                        "labels": [0,1]
                    },
                    {
                        "ratio": 0.2,
                        "labels": [2,3,4,5,6,7,8],
                        "noise": 0.4
                    },
                    {
                        "labels": [9]
                    }
                ]
            }
        ] 
```