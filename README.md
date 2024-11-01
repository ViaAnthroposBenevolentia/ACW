# SIS Graphs Source Code and Results

## Edit and Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ViaAnthroposBenevolentia/ACW/blob/main/SIS3/ColabNotebook.ipynb)

## To Run Locally

1. Clone the repository

```
git clone https://github.com/ViaAnthroposBenevolentia/ACW.git
```

2. Create a virtual environment

```
python -m venv venv
```

3. Activate the virtual environment
```
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate.bat   # In Windows Command Prompt
venv\Scripts\activate.ps1   # In Windows PowerShell
```

4. Install the requirements

```
python -m pip install -r requirements.txt
```

5. Navigate to desired SIS directory and run specific script

```
cd SIS3        # Navigate to SIS3 directory
python step.py # To get the step response graph for example
```
