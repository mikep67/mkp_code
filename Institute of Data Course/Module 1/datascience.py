## *** Import releevant libraries ***

import pandas as pd
import numpy as np
import math
import seaborn as sns
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

%matplotlib inline

### Open the raw data file
### Source: Influenza (laboratory confirmed) Public dataset from Department of Health, Australian Government
### http://www9.health.gov.au/cda/source/pub_influ.cfm

flu_data = pd.read_csv(r"C:\Users\micha\Documents\GitHub\ds-mel-pt-24feb-projects\MichaelPresidente\Project 1\Data\Australian Influenza Public Dataset 2008 to 2017.csv")
