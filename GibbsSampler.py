import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import pandas as pd
from tqdm import tqdm



class GibbsSampler:
    def __init__(self, C_prob: pd.DataFrame, R_prob: pd.DataFrame, S_prob: pd.DataFrame,
                W_prob: pd.DataFrame, e: dict, output: str = None):
        """
        Args:
            C_prob: conditional probability DataFrame for P(C)
            R_prob: conditional probability DataFrame for P(R|C)
            S_prob: conditional probability DataFrame for P(S|C)
            W_prob: conditional probabilityDataFrame for P(W|R, S)
            e: dictionary with evidence variables
            output: path to save Gibbs sampler results as a csv file
        """
        self.C_prob = C_prob
        self.R_prob = R_prob
        self.S_prob = S_prob
        self.W_prob = W_prob
        self.e = e
        self.output = output
        self.samples_matrix = np.empty((0, 4))
    
    def calculate_C_prob(self, current_state: list, C_value: str):
        """
        Args:
            current_state: list of current state of the variables [C, R, S, W]
        Return: 
            probability of passed C value
        """
        nominator = self.C_prob.loc[0, C_value] * self.R_prob.loc[self.R_prob.loc[:, 'C'] == C_value, [current_state[1]]].iloc[0, 0] * self.S_prob.loc[self.S_prob.loc[:, 'C'] == C_value, [current_state[2]]].iloc[0, 0]
        denominator = (self.C_prob.loc[0, 'T'] * self.R_prob.loc[self.R_prob.loc[:, 'C'] == 'T', [current_state[1]]].iloc[0, 0] * self.S_prob.loc[self.S_prob.loc[:, 'C'] == 'T', [current_state[2]]].iloc[0, 0] + 
                        self.C_prob.loc[0, 'F'] * self.R_prob.loc[self.R_prob.loc[:, 'C'] == 'F', [current_state[1]]].iloc[0, 0] * self.S_prob.loc[self.S_prob.loc[:, 'C'] == 'F', [current_state[2]]].iloc[0, 0])
        return nominator/denominator
        
    def calculate_R_prob(self, current_state: list, R_value: str):
        """
        Args:
            current_state: list of current state of the variables [C, R, S, W]
        Return:
            probability of passed R value
        """
        nominator = float(self.R_prob.loc[self.R_prob.loc[:, 'C'] == current_state[0], R_value]) * float(self.W_prob.loc[(self.W_prob.loc[:, 'S'] == current_state[2]) & (self.W_prob.loc[:, 'R'] == R_value), current_state[3]])
        denominator = (float(self.R_prob.loc[self.R_prob.loc[:, 'C'] == current_state[0], 'T']) * float(self.W_prob.loc[(self.W_prob.loc[:, 'S'] == current_state[2]) & (self.W_prob.loc[:, 'R'] == 'T'), current_state[3]]) +
                        float(self.R_prob.loc[self.R_prob.loc[:, 'C'] == current_state[0], 'F']) * float(self.W_prob.loc[(self.W_prob.loc[:, 'S'] == current_state[2]) & (self.W_prob.loc[:, 'R'] == 'F'), current_state[3]]))
        return nominator/denominator

    def calculate_S_prob(self, current_state: list, S_value: str):
        """
        Args:
            current_state: list of current state of the variables [C, R, S, W]
        Return:
            probability of passed S value
        """
        nominator = float(self.S_prob.loc[self.S_prob.loc[:, 'C'] == current_state[0], S_value]) * float(self.W_prob.loc[(self.W_prob.loc[:, 'R'] == current_state[1]) & (self.W_prob.loc[:, 'S'] == S_value), current_state[3]])
        denominator = (float(self.S_prob.loc[self.S_prob.loc[:, 'C'] == current_state[0], 'T']) * float(self.W_prob.loc[(self.W_prob.loc[:, 'R'] == current_state[1]) & (self.W_prob.loc[:, 'S'] == 'T'), current_state[3]]) +
                        float(self.S_prob.loc[self.S_prob.loc[:, 'C'] == current_state[0], 'F']) * float(self.W_prob.loc[(self.W_prob.loc[:, 'R'] == current_state[1]) & (self.W_prob.loc[:, 'S'] == 'F'), current_state[3]]))
        return nominator/denominator
    
    def calculate_W_prob(self, current_state: list, W_value: str):
        """
        Args:
            current_state: list of current state of the variables [C, R, S, W]
        Return:
            sampled value of W
        """
        nominator = self.W_prob.loc[(self.W_prob.loc[:, 'R'] == current_state[1]) & (self.W_prob.loc[:, 'S'] == current_state[2]), W_value]
        denominator = (self.W_prob.loc[(self.W_prob.loc[:, 'R'] == current_state[1]) & (self.W_prob.loc[:, 'S'] == current_state[2]), 'T'] +
                        self.W_prob.loc[(self.W_prob.loc[:, 'R'] == current_state[1]) & (self.W_prob.loc[:, 'S'] == current_state[2]), 'F'])
        return nominator/denominator    
    
    def choose_value(self, current_state: list, calculate_prob):
        """
        Args:
            current_state: list of current state of the variables [C, R, S, W]
            calculate_prob: function to calculate probability of each value
        Return:
            sampled value
        """
        prob_true = calculate_prob(current_state, 'T')
        prob_false = calculate_prob(current_state, 'F')
        return np.random.choice(['T', 'F'], p=[prob_true, prob_false])

    def sample(self, n: int):
        """
        Args:
            n: number of samples to be generated
        """
        variales = ['C', 'R', 'S', 'W']
        x_variables = []
        init_step = []
        for var in variales:
            if var in self.e.keys():
                init_step.append(self.e[var])
            else:
                init_step.append(np.random.choice(['T', 'F']))
                x_variables.append(var)
        # add initial step to samples matrix
        self.samples_matrix = np.append(self.samples_matrix, np.array([init_step]), axis=0)
        # start sampling
        for i in tqdm(range(n)):
            xi = np.random.choice(x_variables)
            current_step = self.samples_matrix[i].copy()
            idx = variales.index(xi)
            if xi == 'C':
                current_step[idx] = self.choose_value(current_step, self.calculate_C_prob)
            elif xi == 'R':
                current_step[idx] = self.choose_value(current_step, self.calculate_R_prob)
            elif xi == 'S':
                current_step[idx] = self.choose_value(current_step, self.calculate_S_prob)
            elif xi == 'W':
                current_step[idx] = self.choose_value(current_step, self.calculate_W_prob)
            self.samples_matrix = np.append(self.samples_matrix, np.array([current_step]), axis=0)
            
        chain = pd.DataFrame(self.samples_matrix, columns=['C', 'R', 'S', 'W']).iloc[1:, :]
        if self.output:
            chain.to_csv(self.output, index=False) 
        return chain


    def sample_modified(self, n: int, burn_in: int, thinning_out: int):
        """
        Args:
            n: number of samples to be generated
        """
        variales = ['C', 'R', 'S', 'W']
        x_variables = []
        init_step = []
        for var in variales:
            if var in self.e.keys():
                init_step.append(self.e[var])
            else:
                init_step.append(np.random.choice(['T', 'F']))
                x_variables.append(var)
        # add initial step to samples matrix
        self.samples_matrix = np.append(self.samples_matrix, np.array([init_step]), axis=0)
        n = burn_in + thinning_out * n
        # start sampling
        for i in tqdm(range(n)):
            xi = np.random.choice(x_variables)
            current_step = self.samples_matrix[i].copy()
            idx = variales.index(xi)
            if xi == 'C':
                current_step[idx] = self.choose_value(current_step, self.calculate_C_prob)
            elif xi == 'R':
                current_step[idx] = self.choose_value(current_step, self.calculate_R_prob)
            elif xi == 'S':
                current_step[idx] = self.choose_value(current_step, self.calculate_S_prob)
            elif xi == 'W':
                current_step[idx] = self.choose_value(current_step, self.calculate_W_prob)
            self.samples_matrix = np.append(self.samples_matrix, np.array([current_step]), axis=0)
            
        chain = pd.DataFrame(self.samples_matrix, columns=['C', 'R', 'S', 'W']).iloc[burn_in+1:, :]
        chain = chain.iloc[::thinning_out, :]
        if self.output:
            chain.to_csv(self.output, index=False) 
        return chain 
    

    @staticmethod
    def estimate_prob(chain: pd.DataFrame, v: str):
        """
        Estimates the marginal probability of varible v.
        Args:
            chain - DataFrame with samples
            v - the name of the variable to be used
        """
        true_count = len(chain.loc[chain.loc[:, v] == 'T', :])
        prob = true_count/len(chain)
        return prob
    

    @staticmethod
    def plot_relative_freq(chains: list, var_list: list, outfile: str, x_lim: int=None):
        """
        Creates and saves plots of the relative frequency of variables in chains.
        Args:
            chains: A list of chains of the same length stored as a DataFrames
            var_list: A list of names of the variables to be plotted
            outfile: Path to save the plot
            x_lim: x-axis limit
        """
        # Updates DataFrames
        for chain in chains:
            chain['step'] = np.arange(1, len(chain)+1)
            for v in var_list:
                chain[f'{v}_true_count'] = 0
                chain[f'{v}_true_count'] = chain[v].apply(lambda x: 1 if x == 'T' else 0).cumsum()
                chain[f'{v}_freq'] = chain[f'{v}_true_count']/chain['step']

        # Creates plots
        fig, axes = plt.subplots(1, len(var_list), figsize=(10*len(var_list), 10))
        for i, v in enumerate(var_list):
            for j, chain in enumerate(chains):
                j += 1
                axes[i].plot(chain['step'], chain[f'{v}_freq'], label=f'sampler run {j}')
            axes[i].legend()
            axes[i].set_title(f'Relative frequency of {v} depending on the number of samples')
            axes[i].set_ylabel(f'Relative frequency of {v}')
            axes[i].set_xlabel('Number of samples')
            if x_lim:
                axes[i].set_xlim(0,x_lim)
        plt.title('Plots of relative frequencies  depending on the number of samples for various runs of the Gibbs sampler')
        plt.savefig(outfile, bbox_inches='tight')


    @staticmethod
    def plot_autocorrelation(chains: list, var_list: list, outfile: str, x_lim: int=100):
        """
        Creates and saves plots of the autocorrelation of variables in chains.
        Args:
            chains: A list of chains of the same length stored as a DataFrames
            var_list: A list of names of the variables to be plotted
            outfile: Path to save the plot
            x_lim: x-axis limit
        """
        # Updates DAtaFrames
        for chain in chains:
            for v in var_list:
                chain[f'coded_{v}'] = chain[v].apply(lambda x: 1.0 if x == 'T' else 0.0)
        
        # Creates plots
        name = outfile.split('.')[:-1]
        name = '.'.join(name)
        for i, sample in enumerate(chains):
            fig, axes = plt.subplots(len(var_list), 1, figsize=(20, 20*len(var_list)))
            k = 0
            for v in var_list:
                x  = sample[f'coded_{v}']
                plot_acf(x, lags=x_lim, ax=axes[k], title=f'Autocorrelation for {v} for the {i+1}. run of the Gibbs sampler')
                axes[k].set_xlabel('Lags')
                axes[k].set_ylabel('Autocorrelation')
                k += 1
            plt.savefig(name+'_'+str(i+1)+'.'+outfile.split('.')[-1], bbox_inches='tight')

    
    @staticmethod
    def gelman_rubin_convergence(chains: list, v: str):
        """
        Funtion to diagnose Gelman & Rubin convergence  
        for variable v in passed chains.
        Args:
            chains: A list of chains of the same length stored as a DataFrames
            v: the name of the variable to be used for the calculation
        """
        # Updates DAtaFrames
        for chain in chains:
            chain[f'coded_{v}'] = chain[v].apply(lambda x: 1.0 if x == 'T' else 0.0)
        # Diagnoses Gelman & Rubin convergence  
        N = len(chains[0])
        M = len(chains)
        means = [np.mean(chain[f'coded_{v}']) for chain in chains]
        sm2s = [chain[f'coded_{v}'].var(ddof=1) for chain in chains] 
        W = np.mean(sm2s)
        B = np.array(means).var(ddof=1)*N
        var_est = (1-1/N)*W + B/N
        R = np.sqrt(var_est/W)

        return R


if __name__ == "__main__":
    # Defining the network
    C_prob = pd.DataFrame({'T': 0.5, 'F': 0.5}, index=[0])
    R_prob = pd.DataFrame({'C': ['T', 'F'], 'T': [0.8, 0.2], 'F': [0.2, 0.8]})
    S_prob = pd.DataFrame({'C': ['T', 'F'], 'T': [0.1, 0.5], 'F': [0.9, 0.5]})
    W_prob = pd.DataFrame({'R': ['T', 'T', 'F', 'F'], 'S': ['T', 'F', 'T', 'F'], 'T': [0.99, 0.9, 0.9, 0.01], 'F': [0.01, 0.1, 0.1, 0.99]})
    evidence = {'S': 'T', 'W': 'T'}
    # Results generating
    # Part 1
    for i in range(10):
        GB = GibbsSampler(C_prob, R_prob, S_prob, W_prob, evidence, output=f'results/task1_100samples_{i}.csv')
        GB.sample(100)

    # Part 2 
    chains = []
    for i in range(2):
        GB = GibbsSampler(C_prob, R_prob, S_prob, W_prob, evidence, output=f'results/task2_50000samples_{i}.csv')
        chains.append(GB.sample(50000))
    GB.plot_relative_freq(chains, ['R', 'C'], 'results/task2_relfreq.png')
    GB.plot_relative_freq(chains, ['R', 'C'], 'results/task2_relfreq_xlim10000.png', x_lim=10000)

    GB.plot_autocorrelation(chains, ['R', 'C'], 'results/task2_autocorr.png', x_lim=49999)
    GB.plot_autocorrelation(chains, ['R', 'C'], 'results/task2_autocorr_xlim200.png', x_lim=200)

    for i in range(10):
        GB = GibbsSampler(C_prob, R_prob, S_prob, W_prob, evidence, output=f'results/task2_100samples_bi2000_to20_{i}.csv')
        GB.sample_modified(100, burn_in=2000, thinning_out=20)