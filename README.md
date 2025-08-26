# The `interpretablefa` package
## Overview
This is a package, which is available on the [Python Package Index](https://pypi.org/project/interpretablefa/), for performing interpretable factor analysis. This implements the priorimax rotation, including the associated interpretability index and plot, that are described [here](https://arxiv.org/abs/2409.11525).

It also contains several helper and visualization functions and wraps several functions from the `factor_analyzer` package.

For more details about the methods implemented and some of the package's features, see [Pairwise Target Rotation for Factor Models](https://arxiv.org/abs/2409.11525). The package's source code can be found [here](https://github.com/interpretablefa/interpretablefa).

The linked paper provides some documentation to the package and docstrings are available. However, a comprehensive guide or documentation is still under development.
## Example
For instance, suppose that `data` contains the dataset, `q` contains the questions, and `p` is the soft constraints matrix. Then, one can fit a 4-factor priorimax model using the snippet below.
```python
import pandas as pd
from interpretablefa import InterpretableFA

# Load the dataset and the questions
data = pd.read_csv("./data/ECR_data_clean.csv")
with open("./data/ECR_questions.txt") as questions_file:
    q = questions_file.read().split("\n")
    questions_file.close()

# Initialize the analyzer
analyzer = InterpretableFA(data, "semantics", q)

# Fit the 4-factor model with the priorimax rotation
analyzer.fit_factor_model("model", 4, "priorimax")

# Get the results
print(analyzer.calculate_indices("model")["v_index"])
print(analyzer.models["model"].rotation_matrix_)

# Visualize the results
analyzer.var_factor_corr_plot("model")
analyzer.interp_plot("model")

```
## Requirements
* Python 3.8 or higher
* `numpy`
* `pandas`
* `scikit-learn`
* `factor_analyzer`
* `tensorflow_hub`
* `scipy`
* `seaborn`
* `matplotlib`
* `statsmodels`
## License
GNU General Public License 3.0
