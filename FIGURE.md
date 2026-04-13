# LaTeX Blocks for `la_zs_ic_ds` Results

Use the blocks below directly in `PROGRESSREPORT.md`.

```latex
\begin{table}[t]
\caption{Policy-level summary metrics for \texttt{la\_zs\_ic\_ds}.}
\label{tab:la-zs-summary}
\centering
\footnotesize
\begin{tabular}{lcccc}
\toprule
\textbf{Policy} & \textbf{Final Reward} & \textbf{Last-20 Avg} & \textbf{ID-A Avg} & \textbf{OOD Avg} \\
\midrule
Nature Policy & 101.00 & 99.30 & 95.78 & -105.16 \\
Adversary Policy & 114.00 & 102.70 & 66.38 & -103.32 \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\caption{ID-A cross-scenario matrix for \texttt{la\_zs\_ic\_ds}.}
\label{tab:la-zs-ida}
\centering
\footnotesize
\begin{tabular}{llcccc}
\toprule
\textbf{Train} & \textbf{Eval} & \textbf{Avg Reward} & \textbf{Collisions} & \textbf{Delivered} & \textbf{Avg Steps} \\
\midrule
nature & nature & 95.78 & 4/50 (0.08/ep) & 46/50 (0.92/ep) & 13.44 \\
nature & adversary & 93.54 & 3/50 (0.06/ep) & 47/50 (0.94/ep) & 16.10 \\
adversary & nature & 92.94 & 4/50 (0.08/ep) & 46/50 (0.92/ep) & 14.52 \\
adversary & adversary & 66.38 & 3/50 (0.06/ep) & 47/50 (0.94/ep) & 24.82 \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\caption{OOD-layout (10 layouts) cross-scenario matrix for \texttt{la\_zs\_ic\_ds}.}
\label{tab:la-zs-ood}
\centering
\footnotesize
\begin{tabular}{llcccc}
\toprule
\textbf{Train} & \textbf{Eval} & \textbf{Avg Reward} & \textbf{Collisions} & \textbf{Delivered} & \textbf{Avg Steps} \\
\midrule
nature & nature & -105.16 & 411/500 (0.82/ep) & 87/500 (0.17/ep) & 43.62 \\
nature & adversary & -113.15 & 419/500 (0.84/ep) & 75/500 (0.15/ep) & 45.52 \\
adversary & nature & -104.29 & 420/500 (0.84/ep) & 79/500 (0.16/ep) & 42.01 \\
adversary & adversary & -103.32 & 405/500 (0.81/ep) & 93/500 (0.19/ep) & 43.87 \\
\bottomrule
\end{tabular}
\end{table}
```

