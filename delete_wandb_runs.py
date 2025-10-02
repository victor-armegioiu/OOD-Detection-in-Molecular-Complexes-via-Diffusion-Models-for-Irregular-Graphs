import re
import wandb

def main():
    entity = "paertschi-eth"
    project = "equivariant-diffusion"

    # Ask user for regex pattern
    pattern = input("Enter regex pattern to match runs: ")
    regex = re.compile(pattern)

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    # Collect matches
    matched_runs = []
    for run in runs:
        if regex.search(run.name):  # change to run.id, run.tags, etc. if needed
            matched_runs.append(run)

    if not matched_runs:
        print("No runs matched.")
        return

    print("Matched runs:")
    for run in matched_runs:
        print(f"- {run.name} (id: {run.id})")

    confirm = input("Delete these runs? (y/N): ").strip().lower()
    if confirm == "y":
        for run in matched_runs:
            run.delete()
        print(f"Deleted {len(matched_runs)} runs.")
    else:
        print("Aborted. No runs deleted.")

if __name__ == "__main__":
    main()
