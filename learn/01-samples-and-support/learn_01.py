"""
# Samples and support

Pipeline check.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1] / "examples"))

# %% [markdown]
# <figure class="bt-figure">
# --8<-- "svg/demo-variogram.svg"
# <figcaption><b>Figure 1.</b> A variogram rises from the nugget to the sill.</figcaption>
# </figure>
#
# The nugget is small here.

# %%
import boitata as bt

print(bt.__version__)
