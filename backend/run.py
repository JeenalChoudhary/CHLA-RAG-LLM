import os
import torch.multiprocessing as mp

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        print("Spawn start method set successfully!")
    except RuntimeError:
        print("Spawn start method already set.")
    gunicorn_args = ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
    os.execvp("gunicorn", gunicorn_args)