# Pipeline Figure (LaTeX)

Use this in `PROGRESSREPORT.md` to visualize the experiment pipeline.

## Preamble Additions

```latex
\usepackage{float} % only needed if you use [H]
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}
```

## Figure Block

```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[
  node distance=4.5mm,
  >=Latex,
  box/.style={draw, rounded corners, align=center, font=\footnotesize, text width=0.90\columnwidth, minimum height=7mm, inner sep=2.5pt},
  io/.style={box, fill=blue!5},
  proc/.style={box, fill=green!5},
  outbox/.style={box, fill=orange!8},
  arr/.style={->, thick}
]

% Vertical single-column flow
\node[io] (default) {\textbf{Base config:} \texttt{configs/default.yaml}};
\node[io, below=of default] (override) {\textbf{Override config:} \texttt{configs/experiments/\textless name\textgreater.yaml}};
\node[proc, below=of override] (runner) {\textbf{Run command:} \texttt{python scripts/run\_experiments.py --only \textless name\textgreater}\\Deep-merge configs and save \texttt{config\_used.yaml}};
\node[proc, below=of runner] (main) {\textbf{Main execution:} \texttt{python -m qlearning\_adversarial.main}\\Comparison mode (train nature + adversary policies)};
\node[proc, below=of main] (eval) {\textbf{Evaluation protocol:} cross-scenario matrix on ID-A and OOD-layout (aggregated over \texttt{ood\_layout\_count})};
\node[outbox, below=of eval] (artifacts) {\textbf{Artifacts under} \texttt{outputs/\textless experiment\_name\textgreater/}:\\
\texttt{csv/} (rewards + matrices), \texttt{learningcurves/}, \texttt{gif/}, \texttt{png/}, \texttt{txt/metrics.txt}, \texttt{pkl/}};

\draw[arr] (default.south) -- (override.north);
\draw[arr] (override.south) -- (runner.north);
\draw[arr] (runner.south) -- (main.north);
\draw[arr] (main.south) -- (eval.north);
\draw[arr] (eval.south) -- (artifacts.north);

\end{tikzpicture}
\caption{Experiment pipeline used in this project. A baseline config is merged with an experiment override, then comparison-mode training and cross-scenario evaluation produce structured artifacts under \texttt{outputs/\textless experiment\_name\textgreater/}.}
\label{fig:pipeline}
\Description{Flow diagram showing configuration merge, experiment execution, policy training, ID-A and OOD evaluation, and generated artifacts including csv, learning curves, gifs, text metrics, and pkl files.}
\end{figure}
```
