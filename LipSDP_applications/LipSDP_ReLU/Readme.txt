The scripts in this Folder Need the LipSDP package to function https://github.com/arobey1/LipSDP
We computed the Lipschitz constants for most networks simply with the command line tool python solve_sdp.py --form network --weight-path ....

Only for the absolute value computation did we write a small Shell script to call the command line function several times. We also inserted the feollowing line into solve_sdp.py to save the results (which would otherwise have been returned as strings)

results = np.append(results, np.array([[L, float(time() - start_time)]]), axis=0)
np.save('absolute_value_results.npy', results)
