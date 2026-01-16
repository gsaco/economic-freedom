# %% [markdown]
# EU SDID main results

# %%
import matplotlib.pyplot as plt

from src.viz.figures import (
    fig_eu_sdid_efw_level_wave2004,
    fig_eu_sdid_reforms_vs_reversals_wave2004,
    fig_macro_tfp_eventstudy_eu,
)

# %%
_ = fig_eu_sdid_efw_level_wave2004()
plt.show()

# %%
_ = fig_eu_sdid_reforms_vs_reversals_wave2004()
plt.show()

# %%
_ = fig_macro_tfp_eventstudy_eu()
plt.show()
