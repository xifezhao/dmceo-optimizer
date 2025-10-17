

---

# Harnessing the Edge of Chaos: A Dynamic Memristive Optimization Algorithm

This repository contains the official Python implementation for the paper **"Harnessing the Edge of Chaos: A Dynamic Memristive Optimization Algorithm for Complex Global Optimization."**

The code provides an implementation of the proposed **Dynamic Memristive Chaos Edge Optimization (DMCEO)** algorithm. It also includes the experimental setup to reproduce all the findings presented in the paper, including benchmarks against four prominent algorithms (PSO, DE, GWO, L-SHADE) and a case study on the Optimal Power Flow (OPF) problem.

---

## 核心成果 | Key Results

### 在基准函数上的性能 | Performance on Benchmark Functions

The results highlight a nuanced performance. While DMCEO is competitive, established algorithms like DE, GWO, and L-SHADE often achieve higher precision on these standard unconstrained problems.

| Function | DMCEO | L-SHADE | GWO | DE | PSO |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ackley (Multimodal)** | 8.95e-01 ± 2.49e+00 | 2.21e-01 ± 4.57e-01 | **4.44e-16 ± 0.00e+00** | 4.47e-15 ± 1.23e-15 | 6.88e+00 ± 8.76e+00 |
| **Griewank (Multimodal)**| 1.47e-02 ± 1.14e-02 | 9.52e-03 ± 9.79e-03 | **0.00e+00 ± 0.00e+00** | **0.00e+00 ± 0.00e+00** | 9.03e+00 ± 2.75e+01 |
| **Rosenbrock (Unimodal)**| 5.82e+01 ± 8.58e+01 | 7.97e-01 ± 1.62e+00 | 2.65e+01 ± 4.19e-01 | **1.04e-02 ± 2.44e-02** | 3.06e+04 ± 4.27e+04 |
| **Sphere (Unimodal)** | 7.99e-02 ± 1.13e-02 | 5.65e-140 ± 1.54e-139| **4.03e-284 ± 0.00e+00**| 1.83e-36 ± 2.00e-36 | 3.33e+02 ± 1.83e+03 |

### 在最优潮流 (OPF) 问题上的性能 | Performance on Optimal Power Flow (OPF)

When applied to the constrained OPF problem, **DMCEO demonstrates its primary strength: exceptional robustness**. It consistently produces high-quality, feasible solutions with a standard deviation orders of magnitude smaller than several competitors.

| Algorithm | Best Fuel Cost ($/h) | Mean Cost | Std Dev |
| :--- | :--- | :--- | :--- |
| PSO | 672,863,217.36 | 672,881,631.18 | 100,856.67 |
| DE | **672,863,217.36** | **672,863,217.36** | **< 1e-5** |
| GWO | 672,863,217.55 | 672,899,571.66 | 114,130.62 |
| L-SHADE | **672,863,217.36** | **672,863,217.36** | **< 1e-5** |
| **DMCEO** | 672,863,217.66 | 672,863,218.89 | **0.75** |

### 收敛曲线 | Convergence Curves

![Convergence Curves for Rosenbrock and Griewank functions](https://storage.googleapis.com/agent-tools-prod.appspot.com/medias/2529/2529513/screenshot_2025-10-16_at_11.02.13_PM.png)

---

## 环境设置 | Setup

### 1. 克隆仓库 | Clone the Repository
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. 创建虚拟环境 (推荐) | Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. 安装依赖 | Install Dependencies

The necessary Python libraries are listed in `requirements.txt`. You can install them using pip.

```bash
pip install -r requirements.txt
```

**`requirements.txt` 内容:**
```
numpy
pandas
tqdm
matplotlib
scipy
pypower
```
*(您可以在您的仓库中创建一个名为 `requirements.txt` 的文件，并将以上内容粘贴进去。)*

---

## 如何运行 | How to Run

The main script `dmceo_experiments.py` is designed to run all experiments and reproduce the results from the paper.

To run the entire suite of experiments, simply execute the script:
```bash
python dmceo_experiments.py
```
*(请将 `dmceo_experiments.py` 替换为您实际的Python脚本文件名。)*

### 预期输出 | Expected Output

The script will:
1.  Run the benchmark function experiments (this may take a significant amount of time).
2.  Save the benchmark results to `results/table_ii_benchmark_summary_standalone.csv`.
3.  Save the convergence plot to `results/figure_2_convergence_curves_standalone.pdf`.
4.  Run the OPF case study (this will also take a long time).
5.  Save the OPF results to `results/table_iii_opf_summary_standalone.csv`.
6.  Print progress and completion messages to the console.

A `results` directory will be created automatically if it does not exist.


---

## 许可证 | License

This project is licensed under the MIT License. See the `LICENSE` file for details.
