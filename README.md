# Question Answer Generation


## Content
1. [Install](#setup) <br>
2. [Train model](#train_model) <br>
    2.1 [Data format](#data_format) <br>
    2.2 [Run train](#run_train) <br
3. [Evaluate model](#evaluate_model) <br>
4. [Model inference](#model_inference) <br>
    4.1 [Using Python Package](#python_package) <br>


## 1. Install <a name="setup"></a>
Run script:
```bash
pip install -r requirements.txt 
pip install -e .
```

## 2. Train model <a name="train_model"></a>
### 2.1 Data format <a name="data_format"></a>
### 2.2 Run train <a name="run_train"></a>
Run script:
```bash
chmod +x run_train.sh
./run_train.sh
```

## 3. Evaluate model <a name="evaluate_model"></a>
## 4. Model inference <a name="model_inference"></a>
### 4.1 Using Python Package <a name="python_package"></a>
```python
from qag_pegasus import QAGPegasus

qag = QAGPegasus(model_name_or_path="mounts/models/gag_pegasus_mrl_model")

context = "Capacitors deviate from the ideal capacitor equation in a number of ways. Some of these, such as leakage current and parasitic effects are linear, or can be assumed to be linear, and can be dealt with by adding virtual components to the equivalent circuit of the capacitor. The usual methods of network analysis can then be applied. In other cases, such as with breakdown voltage, the effect is non-linear and normal (i.e., linear) network analysis cannot be used, the effect must be dealt with separately. There is yet another group, which may be linear but invalidate the assumption in the analysis that capacitance is a constant. Such an example is temperature dependence. Finally, combined parasitic effects such as inherent inductance, resistance, or dielectric losses can exhibit non-uniform behavior at variable frequencies of operation."

outputs = qag.generate_qa(context, num_return_sequences=4)
for sequence in outputs:
    print(sequence)

```

