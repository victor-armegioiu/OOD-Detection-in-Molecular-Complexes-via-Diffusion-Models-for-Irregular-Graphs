import pandas as pd
from scipy.stats import wasserstein_distance


# read pkl file in folder
path = "/cluster/work/math/pbaertschi/benchmarks/PDBbind_train/eval_final/metrics_data.pkl"
# path = "/cluster/work/math/pbaertschi/benchmarks/ours/ours_metrics/metrics_data.pkl"
with open(path, "rb") as f:
    pkl_data = pd.read_pickle(f)

# agg_data = pd.read_csv("/cluster/work/math/pbaertschi/molecular-diffusion/mini_example_dataset_samples/metrics_aggregated.csv")
# det_data = pd.read_csv("/cluster/work/math/pbaertschi/molecular-diffusion/mini_example_dataset_samples/metrics_detailed.csv")

# all_pkl_keys = set([key for entry in pkl_data for key in entry.keys()])
# agg_keys = set(agg_data["metric"])
# det_keys = set(det_data.columns)

# # print all keys 
# # print("All keys in pkl data:", all_pkl_keys)
# # print("All keys in aggregated data:", agg_keys)
# # print("All keys in detailed data:", det_keys)

# for key in all_pkl_keys | agg_keys | det_keys:
#     if "atom" in key.lower():
#         print("Key with 'atom':", key)
#     if "bond" in key.lower():
#         print("Key with 'bond':", key)
#     if "ring" in key.lower():
#         print("Key with 'ring':", key)

# for sublist in pkl_data:
#     for d_key in sublist:
#         if d_key in ["geometry.C-O", "geometry.C=O", "geometry.C-C-O"]:
#             print("Detected", d_key, ": ", sublist[d_key])

# why do geoms produce nan values?
path = "/cluster/work/math/pbaertschi/benchmarks/ours/ours_metrics/samples_distributions.pkl"
with open(path, "rb") as f:
    pkl_data = pd.read_pickle(f)

path2 = "/cluster/work/math/pbaertschi/benchmarks/PDBbind_train/eval_final/trainingPDB_distributions.pkl"
with open(path2, "rb") as f:
    pkl_data2 = pd.read_pickle(f)

for key in pkl_data2.keys():
    if key in ["geometry.C-O", "geometry.C=O", "geometry.C-C-O"]:
        # print(type(pkl_data2[key]))
        print(wasserstein_distance(pkl_data2[key], pkl_data[key]))

# -> top 5 geometries are different between crossdocked and PDBbind training set: 
# ['geometry.C-H', 'geometry.C-C', 'geometry.C-N', 'geometry.C=C', 'geometry.H-N'] ['geometry.C-C-H', 'geometry.H-C-H', 'geometry.C-C=C', 'geometry.C-C-C', 'geometry.C-C-N']
# 




