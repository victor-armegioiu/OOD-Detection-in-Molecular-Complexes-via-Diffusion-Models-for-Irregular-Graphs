import os

# find ../benchmarks/ours/ours_samples/ -maxdepth 2 -type d | wc -l
# grep -R "Sampling .*5b08-A-rec-5b09-4mx-lig-tt-min" logs/

# -----------------------------
# USER CONFIG — EDIT THESE ONLY
# -----------------------------
PARENT_DIR = "../benchmarks/ours3/ours3_samples"

EXPECTED_FOLDERS = {'3u5y-B-rec-3u57-dh8-lig-tt-min', '5mgl-A-rec-5mgl-7mu-lig-tt-min', '5aeh-A-rec-5aeh-8ir-lig-tt-docked', '2e24-A-rec-1j0n-ceg-lig-tt-docked', '3v4t-A-rec-4e7f-udp-lig-tt-min', '3g51-A-rec-3g51-anp-lig-tt-min', '3b6h-A-rec-3b6h-mxd-lig-tt-min', '3u9f-C-rec-3u9f-clm-lig-tt-min', '3tym-A-rec-3n5v-xfh-lig-tt-min', '4aaw-A-rec-4ac3-r83-lig-tt-min', '5liu-X-rec-4gq0-qap-lig-tt-min', '4iiy-A-rec-3tle-gsu-lig-tt-docked', '3daf-A-rec-3daf-feg-lig-tt-docked', '4xli-B-rec-4xli-1n1-lig-tt-min', '2pc8-A-rec-1eqc-cts-lig-tt-docked', '4g3d-B-rec-4idv-13v-lig-tt-min', '2hcj-B-rec-2hcj-gdp-lig-tt-docked', '4zfa-A-rec-1smj-pam-lig-tt-min', '5ngz-A-rec-5ngz-2bg-lig-tt-min', '1jn2-P-rec-1val-png-lig-tt-docked', '4gvd-A-rec-4nxr-ans-lig-tt-min', '1h0i-A-rec-1e6z-ngo-lig-it2-tt', '5q0k-A-rec-5q0q-9ld-lig-tt-docked', '1a2g-A-rec-4jmv-1ly-lig-tt-min', '3nfb-A-rec-3nfb-oae-lig-tt-docked', '4m7t-A-rec-4m7t-sam-lig-tt-min', '4rn0-B-rec-4rn1-l8g-lig-tt-min', '4f1m-A-rec-4f1m-acp-lig-tt-min', '2rma-A-rec-3rdd-ea4-lig-tt-docked', '5b08-A-rec-5b09-4mx-lig-tt-min', '4kcq-A-rec-4cwv-hw8-lig-tt-min', '1r1h-A-rec-1r1h-bir-lig-tt-docked', '4lfu-A-rec-4y13-480-lig-tt-min', '3gs6-A-rec-2oxn-oan-lig-tt-docked', '4keu-A-rec-4ket-pg4-lig-tt-min', '4u5s-A-rec-4u54-3c5-lig-tt-min', '2f2c-B-rec-1xo2-fse-lig-tt-min', '4qlk-A-rec-4qlk-ctt-lig-tt-docked', '1rs9-A-rec-1dmk-itu-lig-tt-min', '2zen-A-rec-2afx-1bn-lig-tt-docked', '5mma-A-rec-4ztf-x2p-lig-tt-min', '2jjg-A-rec-2jjg-plp-lig-tt-min', '4iwq-A-rec-4jlc-su6-lig-tt-min', '5tjn-A-rec-1zj1-nlc-lig-tt-docked', '4rlu-A-rec-4rlu-hcc-lig-tt-min', '3ej8-A-rec-2nsi-itu-lig-tt-min', '4rv4-A-rec-4rv4-prp-lig-tt-docked', '2v3r-A-rec-1dy4-snp-lig-tt-docked', '1umd-B-rec-1umb-tdp-lig-tt-docked', '2e6d-A-rec-2e6d-fum-lig-tt-min', '3li4-A-rec-2gvv-di9-lig-tt-min', '4h3c-A-rec-5cqj-53q-lig-tt-docked', '2gns-A-rec-4qer-stl-lig-tt-min', '4ja8-B-rec-4ja8-1k9-lig-tt-docked', '1phk-A-rec-1phk-atp-lig-tt-min', '4z2g-A-rec-4z2g-m6v-lig-tt-docked', '4tos-A-rec-4tos-355-lig-tt-min', '4yhj-A-rec-4yhj-an2-lig-tt-min', '1coy-A-rec-1coy-and-lig-tt-docked', '1l3l-A-rec-1l3l-lae-lig-tt-min', '2rhy-A-rec-2rhy-mlz-lig-tt-min', '2cy0-A-rec-2d5c-skm-lig-tt-min', '1k9t-A-rec-2wlz-dio-lig-tt-min', '3l3n-A-rec-2iux-nxa-lig-tt-docked', '3kc1-A-rec-3kc1-2t6-lig-tt-min', '4p6p-A-rec-4p77-5rp-lig-tt-docked', '3af2-A-rec-3af4-gcp-lig-tt-min', '3pdh-A-rec-4buz-ocz-lig-tt-min', '1d7j-A-rec-1tco-fk5-lig-tt-docked', '1ai4-A-rec-1ai5-mnp-lig-tt-docked', '14gs-A-rec-20gs-cbd-lig-tt-min', '3pnm-A-rec-3pnq-2ha-lig-tt-docked', '2azy-A-rec-2azy-chd-lig-tt-docked', '3w83-B-rec-2e6d-fum-lig-tt-min', '5w2g-A-rec-5w2i-adp-lig-tt-min', '3jyh-A-rec-3n0t-opy-lig-tt-min', '4azf-A-rec-5lxc-7aa-lig-tt-min', '3o96-A-rec-3o96-iqo-lig-tt-docked', '3dzh-A-rec-3u4i-cvr-lig-tt-docked', '2z3h-A-rec-1wn6-bst-lig-tt-docked', '5d7n-D-rec-4jt9-1ns-lig-tt-min', '1gg5-A-rec-1kbo-340-lig-tt-min', '1djy-A-rec-1djz-ip2-lig-tt-min', '1h36-A-rec-1o79-r23-lig-tt-docked', '5i0b-A-rec-5vef-m77-lig-tt-min', '4bel-A-rec-2ewy-dbo-lig-tt-min', '4tqr-A-rec-2xca-doc-lig-tt-min', '1afs-A-rec-1afs-tes-lig-tt-min', '5bur-A-rec-5x8f-amp-lig-tt-docked', '4pxz-A-rec-4pxz-6ad-lig-tt-min', '2pqw-A-rec-2rhy-mlz-lig-tt-min', '1e8h-A-rec-1e8h-adp-lig-tt-min', '4d7o-A-rec-3n5z-xfm-lig-tt-min', '5l1v-A-rec-5l1v-7pf-lig-tt-docked', '3hy9-B-rec-3hyg-099-lig-tt-min', '4q8b-B-rec-4q8b-sxx-lig-tt-min', '1fmc-B-rec-1fmc-cho-lig-tt-docked', '1dxo-C-rec-1gg5-e09-lig-tt-min', '3chc-B-rec-3ch9-xrg-lig-tt-min', '4aua-A-rec-4aua-4au-lig-it2-tt'}

# -----------------------------


existing = {d for d in os.listdir(PARENT_DIR)
            if os.path.isdir(os.path.join(PARENT_DIR, d))}

missing = sorted(set(EXPECTED_FOLDERS) - existing)

print("\n--- Missing folders ---")
if missing:
    for m in missing:
        print(m)
else:
    print("None")

import os
import re

LOG_DIR = "logs/out/GEN_LIG_48903068"

# regex to match lines like:
# [SHARD 47] Sampling for pocket: 1abc
pattern = re.compile(r"\[SHARD\s+(\d+)\]\s+Sampling for pocket:\s+([A-Za-z0-9_]+)")

processed = []        # list of pocket codes
shard_map = {}        # map shard_id -> list of pockets

for f in os.listdir(LOG_DIR):
    if not (f.endswith(".out") or f.endswith(".log") or f.endswith(".txt")):
        continue

    log_path = os.path.join(LOG_DIR, f)

    try:
        with open(log_path, "r", errors="ignore") as log_file:
            for line in log_file:
                m = pattern.search(line)
                if m:
                    shard_id = int(m.group(1))
                    pocket = m.group(2)

                    processed.append(pocket)
                    shard_map.setdefault(shard_id, []).append(pocket)

    except Exception as e:
        print(f"[WARN] Failed reading {log_path}: {e}")

processed = sorted(set(processed))

print("Total processed pockets:", len(processed))
print("Missing processes", EXPECTED_FOLDERS - set(processed))