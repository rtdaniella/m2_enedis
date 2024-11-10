Dependencies
============

GreenTech Solutions uses a variety of Python packages that are essential for the proper functioning of the application. Below is a detailed explanation of each package included in the `requirements.txt` file, their purpose, and how they contribute to the project.

Packages Overview
-----------------

- **dash**: A web framework used for building the interactive user interface of the application. It allows the creation of web-based dashboards with visualizations, tables, and dynamic content.
  
- **scikit-learn**: A key library for machine learning. It includes various algorithms for classification, regression, and model evaluation, which are used for predicting energy consumption and other related tasks in the application.

- **pandas**: A powerful data manipulation and analysis tool. It is used for data cleaning, transformation, and organizing datasets efficiently.

- **numpy**: This package is used for handling numerical operations, especially for dealing with arrays and matrices. It is fundamental for many operations in data science and machine learning.

- **joblib**: A library used for saving and loading large machine learning models. Since our models are too large to store in the GitHub repository, we use `joblib` to handle model serialization and deserialization.

- **matplotlib**: A comprehensive library for creating static, animated, and interactive visualizations in Python. It is used for plotting graphs and charts to display results.

- **seaborn**: Built on top of matplotlib, it provides a high-level interface for drawing attractive statistical graphics. It is useful for visualizing relationships between variables.

- **scipy**: A library used for scientific computing. It includes modules for optimization, integration, interpolation, eigenvalue problems, and other advanced mathematical functions.

- **imbalanced-learn**: This package provides tools for handling imbalanced datasets, which is essential for our machine learning models to ensure better accuracy and fairness when predicting outcomes.

- **tqdm**: A library used for displaying progress bars, particularly useful when processing large datasets or performing long-running tasks.

- **flask-bootstrap**: Provides a set of Bootstrap components for Flask applications, helping to easily build responsive and visually appealing interfaces.

- **pytest**: A framework that makes it easy to write simple and scalable test cases for our project. We use it to ensure that the application behaves as expected.

How to Install Dependencies
---------------------------

All these dependencies are listed in the `requirements.txt` file.