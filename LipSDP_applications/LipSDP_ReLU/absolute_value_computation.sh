# we write a quick bash script that calls LipSDP to compute the Lipschitz constant estimates
# for 20 different absolute value networks and records the time taken for each computation as well as
# the estimated Lipschitz constants
# finally, we print the mean estimated Lipschitz constant and the mean time taken

# create arrays to store results
estimated_lipschitz_constants=()
times_taken=()

for i in {1..20}; do 
    base_network_path='..\exlipbab_saved_networks\absolute_value_converted\absolute_value_net_(1x6x6x1)_ReLU'
    network_path="${base_network_path}_${i}_weights_converted.mat"
    start_time=$(date +%s.%N)
    
    lipschitz_constant=$(python solve_sdp.py --form neuron --weight-path "$network_path")
    
    end_time=$(date +%s.%N)
    time_taken=$(echo "$end_time - $start_time")
    
    estimated_lipschitz_constants+=($lipschitz_constant)
    times_taken+=($time_taken)
done

# compute mean estimated Lipschitz constant
sum_lipschitz=0
for val in "${estimated_lipschitz_constants[@]}"; do
    sum_lipschitz=$(echo "$sum_lipschitz + $val")
done
mean_lipschitz=$(echo "scale=5; $sum_lipschitz / ${#estimated_lipschitz_constants[@]}")
# compute mean time taken
sum_time=0
for val in "${times_taken[@]}"; do
    sum_time=$(echo "$sum_time + $val")
done
mean_time=$(echo "scale=5; $sum_time / ${#times_taken[@]}")
echo "Mean estimated Lipschitz constant: $mean_lipschitz"
echo "Mean time taken: $mean_time seconds"
