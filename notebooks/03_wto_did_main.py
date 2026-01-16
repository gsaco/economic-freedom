# %% [markdown]
# WTO staggered DiD main results

# %%
import matplotlib.pyplot as plt

from src.viz.figures import fig_dose_response_wto_commitment_depth, fig_wto_did_eventstudy_efw

# %%
_ = fig_wto_did_eventstudy_efw()
plt.show()

# %%
_ = fig_dose_response_wto_commitment_depth()
plt.show()
