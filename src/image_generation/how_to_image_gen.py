# Keep these import statements and add anything else that might be relevant
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import h5py

def run_one_sample(args):
    # Write a function that will run a simulation or create an image
    # (basically do any task) given a set of params in 'args'.
    
    
    # Returns simulation output or image (whatever)
    return None # change this 

# Dont change run_chunk and chunkify
def run_chunk(job_chunk):
    results = []
    for job in job_chunk:
        try:
            results.append(run_one_sample(job))
        except Exception as e:
            print(f"Error processing job {job}: {e}")
            continue
    return results

def chunkify(lst, chunksize):
    for i in range(0, len(lst), chunksize):
        yield lst[i:i + chunksize]

def main():
    # h5file = h5py.File(path, "r+")
    
    # Prepare job list
    # Say you have parameter lists A and B with 10000 samples
    
    # jobs = [(A[i], B[i]) for i in range(len(A))]

    # job_chunks = list(chunkify(jobs, CHUNKSIZE))
    # I always keep chunksize at 2. This can change based on system.

    #total_chunks = len(job_chunks)
    #total_samples = len(jobs)

    # completed_samples = 0
    
    # I always keep number of workers at 6. This can change based on system.
    # Feel free to experiment with both n_workers and chunksize and see which runs 
    # fastest for you on a small number of samples before going large scale. 
    
    # with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # futures = [executor.submit(run_chunk, chunk) for chunk in job_chunks]

        # for future in tqdm(futures, total=total_chunks, desc="Chunks completed"):
          #  results = future.result()
            
           # completed_samples += len(results)
           # tqdm.write(f"Samples done: {completed_samples}/{total_samples}")
            
           # for output in results:
                # do stuff (usually append image to file)
    
    # h5file.close()
    return None # remove this after implementation

if __name__ == "__main__":
    main()