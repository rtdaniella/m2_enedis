Installation
============

To install and run GreenTech Solutions on your local machine, follow these steps.

Prerequisites
-------------

Before you begin the installation, make sure you have the following prerequisites:

1. **Python** (version 3.8 or higher) installed on your machine.
2. **Git** to clone the repository.
3. **A Text Editor or IDE** to work with the code, such as:
   - Visual Studio Code
   - PyCharm
   - Sublime Text
   - Atom

Once these are set up, you can proceed with the installation.

Step 1: Clone the repository
-----------------------------

Start by cloning the repository from GitHub:

git clone https://github.com/rtdaniella/m2_enedis.git cd m2_enedis


Step 2: Create a virtual environment (optional, but recommended)
---------------------------------------------------------------

To avoid conflicts with other projects, create a virtual environment:

python -m venv venv

Activate the virtual environment:

- On Windows:
.'\'venv'\'Scripts'\'activate

- On macOS/Linux:
source venv/bin/activate


Step 3: Install the dependencies
--------------------------------

Use `pip` to install the required packages from the `requirements.txt` file:

pip install -r requirements.txt


Step 4: Add the models
----------------------

Since the models are large, they are not included in the GitHub repository. You can download the models from `here <https://drive.google.com/file/d/1cgaPpuRRpqFje5xGC4d-Vp9CIpt_v8fj/view?usp=sharing>`_ and place them in the `src/utils/models/` directory of the project.

After downloading, the models should be placed in:

m2_enedis/src/utils/models/


Once the models are placed, you can run the application and start using it.

Step 5: Running the application
------------------------------

To start the application, run the following command:

python src/app.py

This will launch the GreenTech Solutions web application.

To see more about the dependencies used in the project, refer to the :doc:`Dependencies </dependencies>`.
