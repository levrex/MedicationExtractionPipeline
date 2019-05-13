# Extracting prescriptions from free text field pipeline
## Notebooks
There are three notebook files:

### 1. MedExtractionPipeline
This notebook file is concerned with the extraction of medication

### 2. ContextPipeline
This notebook file extracts the context according to the rules & targets defined in the corpus directory. The EMR data is merged with Meteor and Medicator information and finally processed. 

### 3. AnalyzeMedicationTrajectory 
This notebook was used for performing statistical analysis

## Functions
### EMR_functions.py
All of the notebooks utilize functions from this file. Every function is documented as well.

## Other files
### acronyms.py
File that is utilized to convert the acronyms to full medication names.

### Visualize.py
Some of the plots are made with this class 

## Graphs directory
The medication trajectory graphs (HTML) are featured in this map

## Bokeh plots directory
The Damerau-Levenshtein distribution is featured in this map

## modules
The majority of necessary modules are featured here

## ra_diagnosis
Features the notebook that was used for the diagnosis extraction
