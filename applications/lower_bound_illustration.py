import numpy as np
import matplotlib.pyplot as plt


# we load the number of subproblems with and withoput lower bounds from the correpoding files
with_lower_bound_subproblems = np.load("wine_net_(11x12x12x1)_ReLU_with_lower_bound_ram_usage_exlipbab.npy")
no_lower_bound_subproblems = np.load("wine_net_(11x12x12x1)_ReLU_no_lower_bound_ram_usage_exlipbab.npy")
print(np.max(np.abs(with_lower_bound_subproblems- no_lower_bound_subproblems)))
fig, ax = plt.subplots(figsize = (9, 2.5))
ax.plot(np.arange(len(with_lower_bound_subproblems)), with_lower_bound_subproblems, label = "with lower bound", linestyle = '--', c = "r", linewidth = 2)
ax.plot(np.arange(len(no_lower_bound_subproblems)), no_lower_bound_subproblems, label = "no lower bound", c = "black", linewidth = 2)


print("number iterations with lower bound", len(with_lower_bound_subproblems))
print("number iterations no lower bound", len(no_lower_bound_subproblems))

ax.legend(fontsize = 15)
ax.set_xlabel("Iteration", fontsize = 15)
ax.set_ylabel("$|\\varrho|$", rotation=0, usetex=True, fontsize = 20)
#set style
fig.tight_layout()
fig.savefig("wine_num_subproblems_illustration.pdf")
plt.show()