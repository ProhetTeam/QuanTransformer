

**Prerequisite**
-Pytorch 1.3+
-Python 3+


**Installation**

```shell
├── quantrans
│   ├── BaseQuanTransformer.py
│   ├── builder.py
│   ├── __init__.py
│   ├── quantops
│   ├── QuanTransformerV1.py
│   ├── QuanTransformerV2.py
│   └── utils
├── README.md
└── setup.py
```



*Step 1: Install QuantQuant*
```shell
git clone https://github.com/ProhetTeam/QuantQuant.git
cd QuantQuant
pip3 install -r requirements.txt
```

*Step 2 : clone the submoddule*
```shell
git submodule init 
git submodule update 
#or git submodule update --init --recursive 

```

*Step 2: Install quantrans & quantops*
```shell
cd QuanTransformer
python3 setup.py install # or pip3 install -v -e . 
```
From then, you can use the different quant operators in your projects.

To verify you can use the quant operators normally, you can test in terminal as follows:
```shell
>>>python
>>>from quantrans.quantops import LSQ
>>>
```
Also, you can use the QuanTransformer() function from quantrans. 
