import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import rosen
from copy import deepcopy

# 忽略数值计算警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 案例研究依赖 ---
from pypower.api import case30, runpf, ppoption
from pypower.polycost import polycost

# ==============================================================================
# SECTION 1: ALGORITHM IMPLEMENTATIONS
# ==============================================================================

class BaseOptimizer:
    """ A base class for all optimizers to ensure a consistent interface. """
    def __init__(self, obj_func, lb, ub, problem_size, epoch, pop_size):
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.problem_size = problem_size
        self.epoch = epoch
        self.pop_size = pop_size
        self.g_best_pos = None
        self.g_best_fit = np.inf
        self.loss_train = []

    def solve(self):
        raise NotImplementedError

    def _update_global_best(self, pop_pos, pop_fit):
        current_best_idx = np.argmin(pop_fit)
        current_best_fit = pop_fit[current_best_idx]
        if current_best_fit < self.g_best_fit:
            self.g_best_fit = current_best_fit
            self.g_best_pos = pop_pos[current_best_idx].copy()
        self.loss_train.append(self.g_best_fit)

class DMCEO(BaseOptimizer):
    """ Implementation of the Dynamic Memristive Chaos Edge Optimization (DMCEO) algorithm. """
    def __init__(self, obj_func, lb, ub, problem_size, epoch, pop_size, **kwargs):
        super().__init__(obj_func, lb, ub, problem_size, epoch, pop_size)
        self.w_range = [0.9, 0.4]
        self.c_range = [2.5, 0.5]
        self.sigma_range = [0.1, 1.0]
        self.M_min, self.M_max = 0.8, 1.2
        self.delta_s, self.delta_f = 0.01, 0.02

    def solve(self):
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
        velocities = np.zeros((self.pop_size, self.problem_size))
        memristive_states = np.random.uniform(self.M_min, self.M_max, self.pop_size)
        chaos_states = np.random.uniform(0, 1, self.pop_size)
        
        p_best_pos = np.copy(positions)
        p_best_fit = np.array([self.obj_func(pos) for pos in p_best_pos])
        stagnation_counters = np.zeros(self.pop_size)
        
        self._update_global_best(p_best_pos, p_best_fit)
        
        for t in range(self.epoch):
            w = self.w_range[0] - t * (self.w_range[0] - self.w_range[1]) / self.epoch
            c = self.c_range[0] - t * (self.c_range[0] - self.c_range[1]) / self.epoch
            sigma = self.sigma_range[0] + t * (self.sigma_range[1] - self.sigma_range[0]) / self.epoch

            for i in range(self.pop_size):
                chaos_states[i] = (4 * memristive_states[i] * chaos_states[i] * (1 - chaos_states[i])) % 1
                r_g = np.random.rand()
                R = np.random.uniform(-1, 1, self.problem_size)
                
                velocities[i] = w * velocities[i] + c * r_g * (self.g_best_pos - positions[i]) + sigma * (chaos_states[i] - 0.5) * R
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)
                
                new_fitness = self.obj_func(positions[i])
                
                if new_fitness < p_best_fit[i]:
                    p_best_pos[i] = positions[i].copy()
                    p_best_fit[i] = new_fitness
                    stagnation_counters[i] = 0
                    memristive_states[i] = max(self.M_min, memristive_states[i] - self.delta_s)
                else:
                    stagnation_counters[i] += 1
                    memristive_states[i] = min(self.M_max, memristive_states[i] + stagnation_counters[i] * self.delta_f)

            self._update_global_best(p_best_pos, p_best_fit)
        return self.g_best_pos, self.g_best_fit, self.loss_train

class OriginalPSO(BaseOptimizer):
    """ Canonical Particle Swarm Optimization. """
    def __init__(self, obj_func, lb, ub, problem_size, epoch, pop_size, **kwargs):
        super().__init__(obj_func, lb, ub, problem_size, epoch, pop_size)
        self.w_range = [0.9, 0.4]
        self.c1, self.c2 = 2.0, 2.0

    def solve(self):
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
        velocities = np.zeros((self.pop_size, self.problem_size))
        p_best_pos = np.copy(positions)
        p_best_fit = np.array([self.obj_func(pos) for pos in p_best_pos])
        
        self._update_global_best(p_best_pos, p_best_fit)

        for t in range(self.epoch):
            w = self.w_range[0] - t * (self.w_range[0] - self.w_range[1]) / self.epoch
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.problem_size), np.random.rand(self.problem_size)
                v_cognitive = self.c1 * r1 * (p_best_pos[i] - positions[i])
                v_social = self.c2 * r2 * (self.g_best_pos - positions[i])
                velocities[i] = w * velocities[i] + v_cognitive + v_social
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)

                new_fitness = self.obj_func(positions[i])
                if new_fitness < p_best_fit[i]:
                    p_best_pos[i] = positions[i].copy()
                    p_best_fit[i] = new_fitness
            
            self._update_global_best(p_best_pos, p_best_fit)
        return self.g_best_pos, self.g_best_fit, self.loss_train

class OriginalDE(BaseOptimizer):
    """ Differential Evolution (DE/rand/1/bin). """
    def __init__(self, obj_func, lb, ub, problem_size, epoch, pop_size, **kwargs):
        super().__init__(obj_func, lb, ub, problem_size, epoch, pop_size)
        self.F = 0.5
        self.CR = 0.9

    def solve(self):
        pop_pos = np.random.uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
        pop_fit = np.array([self.obj_func(pos) for pos in pop_pos])
        
        self._update_global_best(pop_pos, pop_fit)

        for t in range(self.epoch):
            for i in range(self.pop_size):
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x_r1, x_r2, x_r3 = pop_pos[idxs]
                
                mutant_vector = x_r1 + self.F * (x_r2 - x_r3)
                
                j_rand = np.random.randint(0, self.problem_size)
                trial_vector = np.copy(pop_pos[i])
                rand_mask = np.random.rand(self.problem_size) < self.CR
                trial_vector[rand_mask] = mutant_vector[rand_mask]
                trial_vector[j_rand] = mutant_vector[j_rand]
                trial_vector = np.clip(trial_vector, self.lb, self.ub)
                
                trial_fitness = self.obj_func(trial_vector)
                
                if trial_fitness < pop_fit[i]:
                    pop_pos[i] = trial_vector
                    pop_fit[i] = trial_fitness
            
            self._update_global_best(pop_pos, pop_fit)
        return self.g_best_pos, self.g_best_fit, self.loss_train

class OriginalGWO(BaseOptimizer):
    """ Grey Wolf Optimizer. """
    def solve(self):
        pop_pos = np.random.uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
        pop_fit = np.array([self.obj_func(pos) for pos in pop_pos])
        
        sorted_indices = np.argsort(pop_fit)
        alpha_pos, beta_pos, delta_pos = pop_pos[sorted_indices[:3]]
        
        self.g_best_pos = alpha_pos.copy()
        self.g_best_fit = pop_fit[sorted_indices[0]]
        self.loss_train.append(self.g_best_fit)

        for t in range(self.epoch):
            a = 2 - 2 * (t / self.epoch)
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2, self.problem_size)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - pop_pos[i])
                X1 = alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(2, self.problem_size)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * beta_pos - pop_pos[i])
                X2 = beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(2, self.problem_size)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * delta_pos - pop_pos[i])
                X3 = delta_pos - A3 * D_delta

                pop_pos[i] = (X1 + X2 + X3) / 3
            
            pop_pos = np.clip(pop_pos, self.lb, self.ub)
            pop_fit = np.array([self.obj_func(pos) for pos in pop_pos])
            
            sorted_indices = np.argsort(pop_fit)
            alpha_pos, beta_pos, delta_pos = pop_pos[sorted_indices[:3]].copy()
            alpha_fit = pop_fit[sorted_indices[0]]
            
            if alpha_fit < self.g_best_fit:
                self.g_best_fit = alpha_fit
                self.g_best_pos = alpha_pos.copy()
            self.loss_train.append(self.g_best_fit)
        return self.g_best_pos, self.g_best_fit, self.loss_train

class Simplified_LSHADE(BaseOptimizer):
    """ A simplified L-SHADE with core adaptive mechanisms. """
    def __init__(self, obj_func, lb, ub, problem_size, epoch, pop_size, **kwargs):
        super().__init__(obj_func, lb, ub, problem_size, epoch, pop_size)
        self.H = 5  # Memory size
        self.M_F = np.full(self.H, 0.5)
        self.M_CR = np.full(self.H, 0.5)
        self.k = 0
        self.p_best_rate = 0.11

    def solve(self):
        pop_pos = np.random.uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
        pop_fit = np.array([self.obj_func(pos) for pos in pop_pos])
        
        self._update_global_best(pop_pos, pop_fit)

        for t in range(self.epoch):
            S_F, S_CR, delta_fits = [], [], []
            
            for i in range(self.pop_size):
                r_i = np.random.randint(0, self.H)
                F_i = -1
                while F_i <= 0:
                    F_i = np.random.normal(self.M_F[r_i], 0.1)
                F_i = np.clip(F_i, 0, 1)
                
                CR_i = np.random.normal(self.M_CR[r_i], 0.1)
                CR_i = np.clip(CR_i, 0, 1)

                p = int(self.p_best_rate * self.pop_size)
                p = max(2, p)
                best_indices = np.argsort(pop_fit)[:p]
                x_p_best = pop_pos[np.random.choice(best_indices)]
                
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 2, replace=False)
                x_r1, x_r2 = pop_pos[idxs]
                
                mutant_vector = pop_pos[i] + F_i * (x_p_best - pop_pos[i]) + F_i * (x_r1 - x_r2)
                
                j_rand = np.random.randint(0, self.problem_size)
                trial_vector = np.copy(pop_pos[i])
                rand_mask = np.random.rand(self.problem_size) < CR_i
                trial_vector[rand_mask] = mutant_vector[rand_mask]
                trial_vector[j_rand] = mutant_vector[j_rand]
                trial_vector = np.clip(trial_vector, self.lb, self.ub)
                
                trial_fitness = self.obj_func(trial_vector)
                
                if trial_fitness < pop_fit[i]:
                    delta_fit = pop_fit[i] - trial_fitness
                    pop_pos[i] = trial_vector
                    pop_fit[i] = trial_fitness
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    delta_fits.append(delta_fit)
            
            if S_F:
                weights = np.array(delta_fits) / np.sum(delta_fits)
                sum_w_s_f = np.sum(weights * np.array(S_F))
                mean_wL_F = np.sum(weights * np.array(S_F)**2) / sum_w_s_f if sum_w_s_f != 0 else self.M_F[self.k]
                sum_w_s_cr = np.sum(weights * np.array(S_CR))
                mean_wL_CR = np.sum(weights * np.array(S_CR)**2) / sum_w_s_cr if sum_w_s_cr != 0 else self.M_CR[self.k]
                
                self.M_F[self.k] = mean_wL_F
                self.M_CR[self.k] = mean_wL_CR
                self.k = (self.k + 1) % self.H
            
            self._update_global_best(pop_pos, pop_fit)
        return self.g_best_pos, self.g_best_fit, self.loss_train

# ==============================================================================
# SECTION 2: BENCHMARK FUNCTIONS
# ==============================================================================

def sphere_func(x):
    return np.sum(x**2)

def ackley_func(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

def griewank_func(x):
    sum_sq = np.sum(x**2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_sq / 4000 - prod_cos

rosenbrock_func = rosen

# ==============================================================================
# SECTION 3: EXPERIMENTAL VALIDATION SETUP
# ==============================================================================

def run_benchmark():
    print("--- Starting Experimental Validation on Standard Benchmark Suite ---")
    
    benchmark_functions = {
        "Sphere (Unimodal)": {"func": sphere_func, "lb": -100, "ub": 100},
        "Rosenbrock (Unimodal)": {"func": rosenbrock_func, "lb": -30, "ub": 30},
        "Ackley (Multimodal)": {"func": ackley_func, "lb": -32, "ub": 32},
        "Griewank (Multimodal)": {"func": griewank_func, "lb": -600, "ub": 600},
    }
    
    D = 30
    N = 100
    MAX_FES = 10000 * D
    EPOCH = MAX_FES // N
    INDEPENDENT_RUNS = 30
    
    algorithms = {
        "DMCEO": DMCEO, "PSO": OriginalPSO, "DE": OriginalDE,
        "GWO": OriginalGWO, "L-SHADE": Simplified_LSHADE
    }
    
    results = []
    convergence_data = {fname: {algo: [] for algo in algorithms.keys()} for fname in benchmark_functions.keys()}

    for func_name, props in tqdm(benchmark_functions.items(), desc="Benchmark Functions"):
        func_obj = props["func"]
        lb, ub = props["lb"], props["ub"]
        
        for run_id in range(INDEPENDENT_RUNS):
            for algo_name, AlgoClass in algorithms.items():
                model = AlgoClass(func_obj, [lb]*D, [ub]*D, D, EPOCH, N)
                _, best_fit, convergence = model.solve()
                results.append({"Function": func_name, "Algorithm": algo_name, "Run": run_id + 1, "BestError": best_fit})
                convergence_data[func_name][algo_name].append(convergence)

    df_results = pd.DataFrame(results)
    
    summary_stats = df_results.groupby(['Function', 'Algorithm'])['BestError'].agg(['mean', 'std']).reset_index()
    summary_stats['Mean ± Std Dev'] = summary_stats.apply(lambda row: f"{row['mean']:.2e} ± {row['std']:.2e}", axis=1)
    
    table_ii = summary_stats.pivot(index='Function', columns='Algorithm', values='Mean ± Std Dev')
    table_ii = table_ii[["DMCEO", "L-SHADE", "GWO", "DE", "PSO"]]
    
    table_ii.to_csv("results/table_ii_benchmark_summary_standalone.csv")
    print("\nBenchmark results saved to 'results/table_ii_benchmark_summary_standalone.csv'")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    plot_funcs = {"Rosenbrock (Unimodal)": axes[0], "Griewank (Multimodal)": axes[1]}
    for func_name, ax in plot_funcs.items():
        for algo_name, color, marker in [
            ("DMCEO", "C0", "o"), ("PSO", "C1", "s"), ("DE", "C2", "^"),
            ("GWO", "C3", "d"), ("L-SHADE", "C4", "p")
        ]:
            curves = np.array(convergence_data[func_name][algo_name])
            median_curve = np.median(curves, axis=0)
            fes_axis = np.linspace(0, MAX_FES, len(median_curve))
            ax.semilogy(fes_axis, median_curve, label=algo_name, color=color, marker=marker, markevery=int(EPOCH/10), markersize=4)

        ax.set_title(func_name)
        ax.set_xlabel("Function Evaluations (FES)")
        ax.set_ylabel("Fitness Error")
        ax.grid(True, which="both", ls="--")
        ax.set_ylim(1e-9, 1e5)

    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    plt.savefig("results/figure_2_convergence_curves_standalone.pdf", bbox_inches='tight')
    print("Convergence plots saved to 'results/figure_2_convergence_curves_standalone.pdf'")
    plt.close()


# ==============================================================================
# SECTION 4: CASE STUDY SETUP (OPTIMAL POWER FLOW)
# ==============================================================================
def opf_fitness_function_factory(case):
    """ Creates the fitness function for the OPF problem, including penalty handling. """
    gen_buses = case['gen'][:, 0].astype(int)
    bus_indices = case['bus'][:, 0].astype(int)
    pq_buses = np.setdiff1d(bus_indices, gen_buses).astype(int)
    bus_map = {bus_idx: i for i, bus_idx in enumerate(bus_indices)}
    pq_rows = [bus_map[b] for b in pq_buses]
    gen_rows = [bus_map[b] for b in gen_buses]
    
    V_min = case['bus'][pq_rows, 12]
    V_max = case['bus'][pq_rows, 11]
    QG_min = case['gen'][:, 5]
    QG_max = case['gen'][:, 4]
    line_rate_a = case['branch'][:, 5]

    W_V, W_Q, W_S = 1e5, 1e5, 1e5

    def fitness_func(x):
        temp_case = deepcopy(case)
        temp_case['gen'][:, 1] = x[:6]  # Pg
        temp_case['bus'][gen_rows, 7] = x[6:]  # Vg
        
        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        results, success = runpf(temp_case, ppopt)
        
        if not success:
            return 1e12

        Pg = results['gen'][:, 1]  # 发电机有功出力 (MW)
        cost = polycost(temp_case['gencost'], Pg, 1).sum()  # 计算总燃料成本
        
        V_pq = results['bus'][pq_rows, 7]
        V_penalty = np.sum(np.maximum(0, V_pq - V_max)**2) + np.sum(np.maximum(0, V_min - V_pq)**2)
        
        QG = results['gen'][:, 2]
        Q_penalty = np.sum(np.maximum(0, QG - QG_max)**2) + np.sum(np.maximum(0, QG_min - QG)**2)

        from_bus_idx = results['branch'][:, 0].astype(int)
        to_bus_idx = results['branch'][:, 1].astype(int)
        from_rows = [bus_map[b] for b in from_bus_idx]
        to_rows = [bus_map[b] for b in to_bus_idx]

        V_f = results['bus'][from_rows, 7] * np.exp(1j * np.deg2rad(results['bus'][from_rows, 8]))
        V_t = results['bus'][to_rows, 7] * np.exp(1j * np.deg2rad(results['bus'][to_rows, 8]))
        
        branch_z = results['branch'][:, 2] + 1j * results['branch'][:, 3]
        
        S_from = np.abs(V_f * np.conj((V_f - V_t) / branch_z)) * results['baseMVA']
        S_to = np.abs(V_t * np.conj((V_t - V_f) / branch_z)) * results['baseMVA']
        
        S_flow = np.maximum(S_from, S_to)
        active_line_rates = line_rate_a[line_rate_a > 0]
        S_penalty = np.sum(np.maximum(0, S_flow[line_rate_a > 0] - active_line_rates)**2)
        
        return cost + W_V * V_penalty + W_Q * Q_penalty + W_S * S_penalty
    return fitness_func


def run_opf_case_study():
    """Runs the IEEE 30-bus OPF case study."""
    print("\n--- Starting Case Study on IEEE 30-Bus OPF Problem ---")
    
    case = case30()
    opf_fitness_func = opf_fitness_function_factory(case)

    gen_buses = case['gen'][:, 0].astype(int)
    bus_indices = case['bus'][:, 0].astype(int)
    bus_map = {bus_idx: i for i, bus_idx in enumerate(bus_indices)}
    gen_rows = [bus_map[b] for b in gen_buses]
    
    lb = np.concatenate([case['gen'][:, 9], case['bus'][gen_rows, 12]])
    ub = np.concatenate([case['gen'][:, 8], case['bus'][gen_rows, 11]])
    
    D = 12
    N = 100
    MAX_FES = 10000 * D
    EPOCH = MAX_FES // N
    INDEPENDENT_RUNS = 30
    
    algorithms = {
        "DMCEO": DMCEO, "PSO": OriginalPSO, "DE": OriginalDE,
        "GWO": OriginalGWO, "L-SHADE": Simplified_LSHADE
    }
    
    results_opf = []
    
    for run_id in tqdm(range(INDEPENDENT_RUNS), desc="OPF Case Study Runs"):
        for algo_name, AlgoClass in algorithms.items():
            model = AlgoClass(opf_fitness_func, lb, ub, D, EPOCH, N)
            _, best_fit, _ = model.solve()
            results_opf.append({"Algorithm": algo_name, "Run": run_id + 1, "BestCost": best_fit})

    df_opf_results = pd.DataFrame(results_opf)
    
    summary_opf = df_opf_results.groupby('Algorithm')['BestCost'].agg(['min', 'mean', 'std']).reset_index()
    summary_opf = summary_opf.rename(columns={'min': 'Best Fuel Cost ($/h)', 'mean': 'Mean Cost', 'std': 'Std Dev'})
    summary_opf = summary_opf.set_index('Algorithm')
    
    summary_opf = summary_opf.reindex(["PSO", "DE", "GWO", "L-SHADE", "DMCEO"])

    summary_opf.to_csv("results/table_iii_opf_summary_standalone.csv")
    print("\nTable III OPF results saved to 'results/table_iii_opf_summary_standalone.csv'")


# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    np.random.seed(42)  # 设置随机种子以确保可重复性
    if not os.path.exists('results'):
        os.makedirs('results')

    run_benchmark()
    run_opf_case_study()
    
    print("\n--- All experiments completed successfully! ---")