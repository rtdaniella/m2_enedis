src Package Overview
====================

The `src` package serves as the core of the application, containing all essential modules, components, and utilities that power the app's functionality. This package is structured into subpackages to improve modularity and maintainability.

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   src.components
   src.pages
   src.utils

- **src.components**: Contains reusable UI components used across the application, such as navigation bars and footers, to ensure a cohesive user interface.
- **src.pages**: Houses the main application pages, including the home page, prediction pages, and other key views like charts and maps, organized for easy navigation.
- **src.utils**: Provides utility functions and helper classes to support model processing, data handling, and other backend tasks. This includes the `models` module, where machine learning models are trained, saved, and loaded.

Submodules
----------

src.app module
--------------

The `src.app` module initializes and configures the main application, including the setup of routing, layout, and callbacks necessary for interactive functionality.

.. automodule:: src.app
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: src
   :members:
   :undoc-members:
   :show-inheritance:
