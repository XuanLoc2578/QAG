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

qag = QAGPegasus(model_name_or_path="mounts/models/qag_pegasus_mrl_model")

context = '''President Joe Biden ordered airstrikes against Iranian-backed groups in Syria on Tuesday, a little over a week after a number of rockets struck near a military base in northeastern Syria housing US troops. The airstrikes conducted by the US military targeted Iranian-backed groups in Deir ez-Zor Syria, US Central Command said in a statement. The strikes targeted "infrastructure facilities used by groups affiliated with Iran's Islamic Revolutionary Guard Corps," Col. Joe Buccino, a spokesman for CENTCOM, said in the statement. "At President Biden's direction, US military forces conducted precision airstrikes in Deir ez-Zor Syria today. These precision strikes are intended to defend and protect US forces from attacks like the ones on August 15 against US personnel by Iran-backed groups," he said, referring to last week's attacks on the Green Village base near the Iraqi border. The incident did not result in damage or injuries.
'''

outputs = qag.generate_qa(context,
                          num_return_sequences=4,
                          max_length=64,
                          do_sample=True,
                          top_k=6,
                          top_p=0.9,
                          temperature=0.7,
                          no_repeat_ngram_size=2,
                          early_stopping=True
                          )
for sequence in outputs:
    print(sequence)

for sequence in outputs:
    print(sequence)

```

