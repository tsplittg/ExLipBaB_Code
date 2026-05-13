import matplotlib.pyplot as plt
import numpy as np

model_string = f'bike_net_(13x20x16x8x3)_GroupSort_numGroups{10}'

lb_string = f'../exlipbab_saved_networks/bike_sharing/{model_string}_exlipbab_lower_bound_history.npy'
ub_string = f'../exlipbab_saved_networks/bike_sharing/{model_string}_exlipbab_upper_bound_history.npy'



lower_bound_history = np.load(lb_string, allow_pickle=True)
upper_bound_history = np.load(ub_string, allow_pickle=True)



fig, ax = plt.subplots(figsize = (8, 3))

ax.plot(lower_bound_history, label='glb, no initial lower bound', color='blue')
ax.plot(upper_bound_history, label='Global Upper Bound', color='red')
# we set a horizontal line for the initial lower bound
#ax.axhline(y=initial_lower_bound, color='black', linestyle='--', label='New Initial Lower Bound')
ax.set_title(f'ExLipBaB Bound History (Wine data,[11,12,12,1], ReLU)')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Bound Value', fontsize=12)
ax.legend(fontsize = 12)

fig.tight_layout()
fig.savefig(f'../exlipbab_saved_networks/wine/{model_string}_exlipbab_bound_history.pdf')

plt.show()