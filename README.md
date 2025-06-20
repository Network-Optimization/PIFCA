# Iotj - Efficient Federated Clustering with Gradient Search Optimization for Medical Edge Networks

### Experiment Setup

The basic framework used in this experiment is based on **PFLlib**(https://www.pfllib.com/docs.html), with some modifications to the original code as per the references in the paper.

- **Data**:  
  The experiment uses three datasets:  
  - DermaMNIST  
  - BloodMNIST  
  - OrganMNIST

- **Experiment Condition**:  
  The experiments were conducted under the conditions where the Dirichlet distribution of the dataset's classes was set to **0.1**, **1**, and **100**.

### Running the Experiment
Our algorithm PIFCA can perform cluster partitioning on DermaMNIST, OrganAMNIST, and BloodMNIST data through the files `PIFCA-de.ipynb` and `PIFCA-or+bl.ipynb` to obtain clustering results. After obtaining the clustering results, they are passed into `serveravg_test.py`, where `a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]`. The length of `a` should match the number of clients, and users in the same cluster should be assigned the same cluster label. The baseline algorithms can be directly executed by running the `sh1-1.py` file.
The file has already stored datasets under different data distributions.
To run the experiment, use the following command:
```bash

python main1.py -data OrganAMNIST_0.1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1   # using OrganAMNIST dataset

Alternatively, you can directly write the experiments you want to run in the sh1-1.py file in the format described above, and execute the file to perform the multi-threaded experiments.
