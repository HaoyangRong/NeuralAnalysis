# NeuralAnalysis
 
This repo contains some of the files from a yet to be published work, mostly some tools I developed for this project, which belong to a larger pipeline.

**Data.py**

It contains the classes I built for easy, fool-proof data retrieval and manipulation.
The Loader class creates a loader object, from which the data sets can be queried very flexibly using regular expressions, through member function get. Note the “lazy loading” principle was applied in the design. No data set will present in memory until it’s queried.


**StdImports.py**

I repeatedly use some packages in different analysis scripts. This file is used as a “header file” to include those packages and some “global” variables. Therefore, the analysis scripts can be kept clean. And prevent the case that one important variable is changed in one file, but forget to do so in other files.


More readme contents are under construction :construction: :smiley: