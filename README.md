 # GreenTech Solutions
 
  GreenTech Solutions is a web application built with **Dash**, developed as part of a university project for Master 2 SISE students (Statistics and Computer Science for Data Science). The project is supervised by M. Sardellitti Anthony. The purpose of this application is to analyze and study the **Energy Performance Diagnosis (DPE)** of buildings in the **Rhône department (69)**, as well as to predict the total energy consumption of buildings. The data used comes from the **public API of the dataset: DPE Existing Buildings (since July 2021)**, available on the open data portal of [ADEME] (https://data.ademe.fr/datasets/dpe-v2-logements-existants).

  
 ### Project Objectives:

 1. **Analysis of DPE and buildings**:
 
  Analyze the characteristics of buildings (construction year, size, etc.) and energy performance (DPE) to identify factors influencing energy consumption. 
  
 2. **DPE Label Prediction Model**:
 
  Build a model to predict the DPE label of a building (A, B, C, D, E or F.) based on its characteristics. 
 
 3. **Energy Consumption Prediction Model**:
 
  Develop a model to predict the total final energy consumption of a building, in kWh/hef/year, for the considered energy type. The application uses **regression** and **classification** models to perform these predictions and integrates interactive visualizations such as **maps** and **charts** to make the analysis more accessible and user-friendly. 

  
  ## Table of Contents 

  - [Installation](#installation)
  - [Usage](#usage)
  - [Dependencies](#dependencies) 
  - [Models](#models) 
  - [Contributing](#contributing) 
  - [License](#license)

   --- 

   ## Installation To install and run GreenTech Solutions locally, follow the steps below.
   
   ### Prerequisites 
   - Python 3.8 or higher 
   - A code editor like [VS Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/) 
   - A virtual environment (recommended) 
   
   ### Steps 
   
   1. Clone the repository:
   
    ```bash git clone https://github.com/rtdaniella/m2_enedis.git cd m2_enedis ``` 
    
   2. Create a virtual environment (optional but recommended): 
   
   ```bash python3 -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate ``` 
   
   3. Install the required dependencies from `requirements.txt`: ```bash pip install -r requirements.txt ``` 
   
   4. Download the large machine learning models and place them in the `src/utils/models/` directory. The models can be downloaded from [here](https://drive.google.com/file/d/1cgaPpuRRpqFje5xGC4d-Vp9CIpt_v8fj/view?usp=sharing). 
   
   5. Run the application:
   
    ```bash python src/app.py ``` 
    
   6. Open your browser and visit `http://127.0.0.1:8050` to see the app in action. 
   
   --- 
   
   ## Usage 
   
   Once installed, you can start interacting with the application. The key features include: 
   
   - **Prediction of energy consumption**: Input details about a building to predict its energy consumption using the pre-trained machine learning models. 
   
   - **DPE Prediction**: Predict the Energy Performance Diagnosis (DPE) for a property. 
   
   - **Interactive map**: Visualize and explore data about buildings and their energy profiles on a map of the Rhône (69) department. 
   
   - **Charts and graphs**: Explore various insights through interactive charts and graphs. 
   
   --- 
   
   ## Dependencies GreenTech Solutions relies on several Python packages. The list of dependencies is provided in the `requirements.txt` file, which includes popular packages such as: 
   
   - **Dash**: For building interactive web applications. 
      
   - **Plotly**: For interactive plots and graphs. 
   
   - **Scikit-learn**: For machine learning models. 
   
   - **Pandas, NumPy**: For data manipulation and processing. To install the dependencies, simply run:
   
    ```bash pip install -r requirements.txt ``` 
   
   --- 
   
   ## Models 
   
   Since the machine learning models are quite large, they are not stored in the GitHub repository. You can download the models from the following link and place them in the `src/utils/models/` directory:
    
   - [Download models from Google Drive](https://drive.google.com/file/d/1cgaPpuRRpqFje5xGC4d-Vp9CIpt_v8fj/view?usp=sharing) 
   
  
   ---
   
   ## Documentation

   For a more detailed explanation of the application's functionality and technical aspects, please refer to the [Technical Documentation](https://drive.google.com/drive/folders/1DT6DHTjo8yRGUAuXK9Bm5FodsxceZ3RU?usp=sharing).
   
   ---
   
   ## Credits

   This application was developed by:

    - Daniella Rakotondratsimba
    - Akrem Jomaa
    - Béranger Thomas
      
   
   ---

   ## License GreenTech Solutions is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.