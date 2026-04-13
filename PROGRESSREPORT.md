%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% LaTeX Template for AAMAS-2026 (based on sample-sigconf.tex)
%%% Prepared by the AAMAS-2026 Publication Chairs based on the version from AAMAS-2025. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Start your document with the \documentclass command.


%%% == IMPORTANT ==
%%% Use the first variant below for the final paper (including author information).
%%% Use the second variant below to anonymize your submission (no author information shown).
%%% For further information on anonymity and double-blind reviewing, 
%%% please consult the call for paper information
%%% https://cyprusconferences.org/aamas2026/submission-instructions/

%%%% For anonymized submission, use this
% \documentclass[sigconf,anonymous]{aamas} 

%%%% For camera-ready, use this
\documentclass[sigconf]{aamas} 


%%% Load required packages here (note that many are included already).

\usepackage{balance} % for balancing columns on the final page
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}
\usepackage{float}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% AAMAS-2026 copyright block (do not change!)

\setcopyright{ifaamas}
\acmConference[AAMAS '26]{Proc.\@ of the 25th International Conference
on Autonomous Agents and Multiagent Systems (AAMAS 2026)}{May 25 -- 29, 2026}
{Paphos, Cyprus}{C.~Amato, L.~Dennis, V.~Mascardi, J.~Thangarajah (eds.)}
\copyrightyear{2026}
\acmYear{2026}
\acmDOI{}
\acmPrice{}
\acmISBN{}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% == IMPORTANT ==
%%% Use this command to specify your submission number.
%%% In anonymous mode, it will be printed on the first page.

% \acmSubmissionID{<<submission id>>}

%%% Use this command to specify the title of your paper.

\title{Adversarial Q-Learning in Warehouse Environments}

%%% Provide names, affiliations, and email addresses for all authors.

\author{Luke Stockbridge}
\affiliation{
  \institution{Washington University}
  \city{St. Louis}
  \country{United States}}
\email{l.d.stockbridge@wustl.edu}

\author{Yubin Xu}
\affiliation{
  \institution{Washington University}
  \city{St. Louis}
  \country{United States}}
\email{yubin@wustl.edu}

%%% Use this environment to specify a short abstract for your paper.

\begin{abstract}
In this paper we explore Q-Learning in a grid world warehouse environment. We ultimately seek to determine if an agent that trains against a true adversary generalizes better to out of distribution settings than an agent that trains only against stochastic obstacles. 

We examine many different spin-offs of this question to determine in what scenarios an adversary makes the agent more robust. For example we explore both deterministic and learned adversaries as well as generalization to new layouts and new obstacle intensities. We explore different configurations and their impact on the agent's observed behavior such as implementing distance shaping heuristics and sweeping through Q-Learning parameters.

As this is a progress report and not a final result, we will mainly focus on demonstrating progress towards our final goal (the challenges faced so far and the lessons we have learned) rather than directly answering and backing up our overarching question. These challenges mainly revolve around determining what combination of factors are worth varying/exploring both in terms of which factors could produce meaningful real-world interpretations and achieving desirable agent behavior. 
\end{abstract}

%%% Use this command to specify a few keywords describing your work.
%%% Keywords should be separated by commas.

\keywords{Q-Learning, Adversary, Agents}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Include any author-defined commands here.
         
\newcommand{\BibTeX}{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em\TeX}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{document}

%%% The following commands remove the headers in your paper. For final 
%%% papers, these will be inserted during the pagination process.

\pagestyle{fancy}
\fancyhead{}

%%% The next command prints the information defined in the preamble.

\maketitle 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Introduction}

Warehouse automation environments are incredibly dynamic and even chaotic in practice. Increasingly popular delivery robots must reach a package, pick it up, and deliver it while operating around traffic and disruptions (e.g., moving forklifts, shelving, blocked aisles). The central problem in this project is not only whether an RL policy can solve one fixed environment, but whether a policy trained under one disturbance regime transfers robustly to unseen layouts and disturbance realizations.

Motivated by robust/adversarial RL and dynamic-warehouse navigation literature \cite{pinto2017rarl,tessler2019actionrobust,kristiansson2025warehouse}, we frame our work as a controlled comparison between two training strategies in the same gridworld task: (i) training against stochastic disturbances (``nature''), and (ii) training against an active adversary (``adversary''). We evaluate both trained policies under identical protocols with in-distribution evaluation (ID-A: same base layout family as training) and out-of-distribution evaluation (OOD-layout: unseen sampled layouts).

This progress report focuses on the current system state, major implementation decisions, and interim findings rather than final claims. Concretely, we now have a reproducible experiment pipeline with configuration-driven runs, per-experiment artifact management, cross-scenario evaluation matrices, and rollout visualizations. We also implemented and tested several extensions intended to improve robustness, including distance shaping, richer state context (relative package/destination features), learned adversary behavior, and a zero-sum adversary objective.

The main takeaway at this stage is that we have substantially improved infrastructure and diagnostic visibility, but the scientific question remains open: adversarial training has not yet produced a clear, stable OOD advantage over stochastic training in our tabular setting. The remainder of the report details what has been built, what has been learned, and which targeted next steps are most likely to answer the robustness question convincingly.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Definitions and Experimental Setup}

\subsection{Task Definition}
We model a warehouse as a discrete $5\times5$ gridworld. The primary learning agent must complete a pickup-and-delivery task: navigate to a package location, pick up the package, and then deliver it to a designated destination cell while minimizing cumulative cost. The environment includes static shelves (blocked cells) and dynamic hazards, and an episode terminates either on successful delivery, collision with a terminal hazard, or timeout.

At each step, the agent selects one of four cardinal actions (up, right, down, left). Rewards combine movement cost and task progress:
\begin{itemize}
    \item negative per-step cost and obstacle penalties,
    \item positive pickup and delivery rewards,
    \item optional shaping based on distance-to-objective progress.
\end{itemize}
This reward structure makes the objective ``deliver quickly and safely'' rather than simply ``eventually deliver.''

\begin{figure}[H]
  \centering
  \fbox{\parbox{0.93\linewidth}{
  \textbf{Reward configuration used in experiments.}

  \vspace{0.5em}
  \begin{itemize}\itemsep0pt \parskip0pt \parsep0pt
    \item Step penalty: \texttt{-2.0}
    \item Obstacle penalty: \texttt{-3.0}
    \item Adversary/Forklift penalty: \texttt{-50.0}
    \item Pickup reward: \texttt{25.0}
    \item Delivery reward: \texttt{100.0}
    \item Distance shaping: \texttt{disabled} (\texttt{scale = 0.0})
  \end{itemize}
  }}
  \caption{Reward and penalty values used for training and evaluation.}
  \label{fig:rewards}
  \Description{Boxed list of reward and penalty values: step penalty -2.0, obstacle penalty -3.0, forklift penalty -50.0, adversary penalty -50.0, pickup reward 25.0, delivery reward 100.0, distance shaping disabled with scale 0.0.}
\end{figure}

\begin{figure}[H]
  \centering
  % Put your image file in the Overleaf project (e.g., under figs/)
  \includegraphics[width=0.85\linewidth]{env-example.png}
  \caption{Example warehouse gridworld layout for the adversary scenario. Colors indicate the start cell, package location, destination, shelf (blocked) cells, and the adversary position.}
  \label{fig:env-map-example}
  \Description{A 5x5 gridworld visualization showing a start cell in blue, a package in yellow, a destination in green, shelf obstacles in dark gray, and an adversary in magenta, with a legend.}
\end{figure}


\subsection{Disturbance Regimes}
We evaluate two disturbance regimes that define the training world:
\begin{itemize}
    \item \textbf{Nature scenario:} static shelves with stochastic `forklift' hazards.
    \item \textbf{Adversary scenario:} static shelves with an active adversary hazard.
\end{itemize}

The adversary is configurable as:
\begin{samepage}
\begin{itemize}
    \item \textbf{Deterministic pursuit} (greedy Manhattan-distance minimization with random tie-breaks), or
    \item \textbf{Learning adversary} (tabular learner with either heuristic objective or strict zero-sum objective against the delivery agent).
\end{itemize}
\end{samepage}
One of our current comparisons focuses on how the \emph{training regime} affects robustness of the delivery policy.

\subsection{Policy Comparison Protocol}
For each run in comparison mode, we train two separate delivery policies:
\begin{enumerate}
    \item a policy trained in the nature scenario;
    \item a policy trained in the adversary scenario.
\end{enumerate}
Both are then evaluated in a cross-scenario matrix (nature-on-nature, nature-on-adversary, adversary-on-nature, adversary-on-adversary) under:

\begin{itemize}
    \item \textbf{ID-A:} same base layout family used during training;
    \item \textbf{OOD-layout:} unseen sampled layouts (aggregated across multiple layouts).
\end{itemize}

We report average return, collision count, delivered packages, and average steps per episode.  To complement endpoint metrics, we also track reward-learning dynamics over training (see Figure~\ref{fig:learning-curves-placeholder}).

\begin{figure}[t]
  \centering
  % Replace filename with your exported learning-curves image.
  \includegraphics[width=0.85\linewidth]{learning-curves.png}
  \caption{Policy learning curve of reward trajectory for the nature-trained agent.}
  \label{fig:learning-curves-placeholder}
  \Description{Policy learning curve of reward trajectory for the nature-trained agent.}
\end{figure}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Infrastructure and Experiment Pipeline}

Our experimentation workflow is configuration-driven and designed to support controlled comparisons with minimal manual code edits. The repository uses a baseline configuration (\path{configs/default.yaml}) and experiment-specific override files (\path{configs/experiments/*.yaml}). At run time, the override is deep-merged onto the default configuration, and the exact merged configuration is saved alongside outputs as \path{config_used.yaml}. This is important for reproducibility because every reported result is tied to an explicit, versioned set of parameters.

\subsection{Run Workflow}
The primary execution path is:
\begin{itemize}
    \item create or edit an override file in \path{configs/experiments/},
    \item run experiment script with \path{ --only <experiment_name>},
    \item collect outputs from \path{outputs/<experiment_name>/}.
\end{itemize}

In comparison mode, one command trains two separate delivery policies (nature-trained and adversary-trained), then evaluates both policies on both scenarios for ID-A and OOD-layout. OOD-layout metrics are aggregated over \path{eval.ood_layout_count} sampled layouts (default currently set to 5 in baseline runs).

\subsection{Configurable Knobs in Current Pipeline}
The current infrastructure exposes the following tunable groups:
\begin{itemize}
    \item \textbf{Core run budget:} training/evaluation episode counts
    \item \textbf{Agent learning:} $\alpha$, $\gamma$, epsilon schedule, shared vs non-shared Q-table (if testing multiple delivery agents).
    \item \textbf{Task/reward model:} step cost, obstacle penalties, hazard penalties, pickup/delivery rewards, distance shaping.
    \item \textbf{State representation:} optional relative package/destination context features.
    \item \textbf{Scenario dynamics:} forklift movement probability, adversary policy type (deterministic vs learned), adversary move probability, move budget, and adversary-learning hyperparameters/objective (heuristic vs zero-sum).
    \item \textbf{Evaluation controls:} ID-A/OOD protocol parameters and OOD layout count.
\end{itemize}

\subsection{Artifacts and Diagnostics Collected Per Run}
Each experiment directory contains structured outputs that support quantitative and qualitative analysis:
\begin{itemize}
    \item \texttt{csv/}: training rewards and cross-scenario matrices,
    \item \texttt{learningcurves/}: reward trajectories over episodes,
    \item \texttt{gif/}: rollout animations for ID-A and OOD cross-scenario pairs,
    \item \texttt{png/}: static layout snapshots,
    \item \texttt{txt/metrics.txt}: compact run summary (including return, collisions, deliveries, and average steps),
    \item \texttt{pkl/}: serialized policy tables and environment configs for replay/debugging.
\end{itemize}

This infrastructure has been essential for debugging behavior-level issues (e.g., no-move policies, tie-break artifacts, adversary learning pathologies) and systematically testing different configurations.

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Interim Experiments and Lessons Learned}
We have run a sequence of controlled experiments that vary disturbance type, reward shaping, state context, and adversary objective.

\subsection{Nature Baseline Reference}
To ground the comparisons, we first note the delivery-policy performance when trained and evaluated in the nature setting. In the baseline run (\texttt{outputs/default}), the nature-trained policy achieves strong ID-A performance (\texttt{id\_a\_nature\_on\_nature\_reward = 86.80}, \texttt{47/50} deliveries, \texttt{3/50} collisions), but poor OOD transfer (\texttt{ood\_layout\_nature\_on\_nature\_reward = -433.67}, \texttt{28/250} deliveries). This provides an initial reference point for interpreting whether adversary-based training actually improves robustness beyond stochastic-disturbance training.

\subsection{Deterministic vs Learned Adversary}
Initially we wanted to determine how our adversary should behave. Our baseline deterministic-adversary run (\path{outputs/default}) produced strong same-scenario ID-A performance for the adversary-trained policy (\path{id_a_adversary_on_adversary_reward = 94.40}), but poor OOD robustness (\path{ood_layout_adversary_on_adversary_reward = -101.51}).


When we switched to a learned adversary (\texttt{outputs/\allowbreak learning\allowbreak adversary}), adversary-trained performance dropped substantially in both ID-A and OOD (\texttt{33.84} and \texttt{-119.32}, respectively). This indicates that, in the current tabular setup, a learned adversary does not yet provide stronger or cleaner training pressure than deterministic pursuit. However, we want to attempt to move forward with a learned adversary. Our ultimate goal is to train against an adversary that learns to and actively tries to produce difficult circumstances for the agent. 

\subsection{Distance Shaping and Increased Context}
Distance shaping alone (\texttt{outputs/distance\_shaping}) improved ID-A in adversary-on-adversary (\texttt{108.66}), but did not resolve the OOD falloff in performance.  

Adding relative package/destination context without shaping (\texttt{outputs/inc\_context}) produced brittle behavior, including degenerate OOD outcomes (e.g., \texttt{-400.00} with zero deliveries in some cells), consistent with sparse coverage in an expanded state space.  

Combining context + shaping (\texttt{outputs/\allowbreak inc\_\allowbreak context\_\allowbreak distance\_\allowbreak shaping}) gave the most promising OOD trend, improving adversary-on-adversary OOD to \texttt{-61.91} (vs \texttt{-101.51} baseline), though still far from robust.

\subsection{General-Sum vs Zero-Sum Learned Adversary}
After reviewing the animations of evaluation episodes, we noticed the adversarial agent frequently did not move or take action that indicated it had a goal of interfering with the agent. This led to us adjusting adversary learning parameters (e.g. more discovery actions early on to decrease context sparseness) and experimenting with a zero-sum game.

For learned-adversary experiments with context + shaping:
\begin{itemize}
    \item \textbf{General-sum} variant (\texttt{outputs/\allowbreak la\_\allowbreak ic\_\allowbreak ds}) achieved
    \texttt{id\_\allowbreak a\_\allowbreak adversary\_\allowbreak on\_\allowbreak adversary = 91.42} and
    \texttt{ood\_\allowbreak layout\_\allowbreak adversary\_\allowbreak on\_\allowbreak adversary = -63.61}.
    \item \textbf{Zero-sum} variant (\texttt{outputs/\allowbreak la\_\allowbreak zs\_\allowbreak ic\_\allowbreak ds}) used a larger training budget (20k episodes) and 10 OOD layouts. It achieved
    \texttt{id\_\allowbreak a\_\allowbreak adversary\_\allowbreak on\_\allowbreak adversary = 66.38} and
    \texttt{ood\_\allowbreak layout\_\allowbreak adversary\_\allowbreak on\_\allowbreak adversary = -103.32}.
\end{itemize}

In the zero-sum run, low ID-A collisions (\texttt{3/50}) and high deliveries (\texttt{47/50}) coexisted with lower return because average episode length was much larger (\texttt{24.82} steps), highlighting that efficiency penalties dominate reward when trajectories are long. Once again, though with our current configuration this decreases performance, it seems to produce much more reasonable adversary behavior. We believe that it is possible that through additional experimentation and tinkering with configuration, having a more difficult (but constrained) adversary could benefit the robustness of the learned policy.

\subsection{Lessons and Current Failure Modes}
\begin{itemize}
    \item \textbf{OOD generalization remains the dominant weakness.} Across all variants, OOD returns are strongly negative and collision rates remain high.
    \item \textbf{Learned adversary is not yet reliably stronger than deterministic pursuit.} Current adversary-learning dynamics are sensitive to objective and hyperparameters.
    \item \textbf{State enrichment helps only with sufficient learning signal.} Relative context without shaping can hurt due to state explosion; context + shaping is better but still not sufficient.
    \item \textbf{Reward must be interpreted with steps.} We added average-step reporting because return alone hides path inefficiency effects.
    \item \textbf{Zero-sum improves pressure, not OOD robustness.} The adversary behaves more anti-agent, but OOD returns remain strongly negative.
\end{itemize}

\begin{table}[t]
\caption{Representative same-scenario results from completed runs.}
\label{tab:interim-core}
\centering
\footnotesize
\begin{tabular}{lcc}
\toprule
\textbf{Experiment} & \textbf{ID-A Adv$\rightarrow$Adv} & \textbf{OOD Adv$\rightarrow$Adv} \\
\midrule
\texttt{default} & 94.40 & -101.51 \\
\texttt{learningadversary} & 33.84 & -119.32 \\
\texttt{la\_ic\_ds} & 91.42 & -63.61 \\
\texttt{la\_zs\_ic\_ds} & 66.38 & -103.32 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Example Run Statistics and Cross-Matrix Output}
To make the reporting format concrete, we include one full example from \texttt{la\_zs\_ic\_ds}. Table~\ref{tab:la-zs-summary} shows the immediate policy-level summary printed at the end of a run. Tables~\ref{tab:la-zs-ida} and~\ref{tab:la-zs-ood} show the ID-A and OOD cross-scenario matrices. This is the first place in the report where the full cross-matrix structure is shown explicitly, rather than only referenced in text.

\begin{table}[H]
\caption{Policy-level summary metrics for \texttt{la\_zs\_ic\_ds}.}
\label{tab:la-zs-summary}
\centering
\footnotesize
\begin{tabular}{lcccc}
\toprule
\textbf{Policy} & \textbf{Final} & \textbf{Last-20} & \textbf{ID-A} & \textbf{OOD} \\
\midrule
Nature Policy & 101.00 & 99.30 & 95.78 & -105.16 \\
Adversary Policy & 114.00 & 102.70 & 66.38 & -103.32 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\caption{ID-A cross-scenario matrix for \texttt{la\_zs\_ic\_ds} (50 eval episodes).}
\label{tab:la-zs-ida}
\centering
\footnotesize
\begin{tabular}{lcccc}
\toprule
\textbf{Train$\rightarrow$Eval} & \textbf{Avg Reward} & \textbf{Coll./ep} & \textbf{Deliv./ep} & \textbf{Avg Steps} \\
\midrule
nature$\rightarrow$nature & 95.78 & 0.08 & 0.92 & 13.44 \\
nature$\rightarrow$adversary & 93.54 & 0.06 & 0.94 & 16.10 \\
adversary$\rightarrow$nature & 92.94 & 0.08 & 0.92 & 14.52 \\
adversary$\rightarrow$adversary & 66.38 & 0.06 & 0.94 & 24.82 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\caption{OOD-layout cross-scenario matrix for \texttt{la\_zs\_ic\_ds} (10 layouts, 500 eval episodes total).}
\label{tab:la-zs-ood}
\centering
\footnotesize
\begin{tabular}{lcccc}
\toprule
\textbf{Train$\rightarrow$Eval} & \textbf{Avg Reward} & \textbf{Coll./ep} & \textbf{Deliv./ep} & \textbf{Avg Steps} \\
\midrule
nature$\rightarrow$nature & -105.16 & 0.82 & 0.17 & 43.62 \\
nature$\rightarrow$adversary & -113.15 & 0.84 & 0.15 & 45.52 \\
adversary$\rightarrow$nature & -104.29 & 0.84 & 0.16 & 42.01 \\
adversary$\rightarrow$adversary & -103.32 & 0.81 & 0.19 & 43.87 \\
\bottomrule
\end{tabular}
\end{table}


\section{Open Challenges and Next Steps}

Our main open challenge is still OOD robustness. Even when ID-A metrics are strong, performance drops sharply under unseen layouts. Additionally, learned-adversary behavior remains inconsistent across objectives and hyperparameters.

Near-term next steps are:
\begin{itemize}
    \item \textbf{Stabilize learned adversary training:} run focused sweeps for adversary epsilon schedule, move probability, and objective settings (heuristic vs zero-sum).
    \item \textbf{Improve protocol reliability:} repeat key experiments across multiple seeds and report mean/variance rather than single-run outcomes.
    \item \textbf{Target OOD directly:} continue to run new experiments and compare matched-budget runs under the same OOD-layout count.
    \item \textbf{Increase environment complexity:} consider holding steps constant and measuring throughput (packages delivered per episode before max steps or collision occurs), adding multiple delivery agents, or increasing static 'shelving' obstacles
\end{itemize}

These steps are intended to answer the central question more rigorously: when, if ever, adversarial training improves generalization beyond stochastic-disturbance training.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{thebibliography}{9}

\bibitem{pinto2017rarl}
L.~Pinto, J.~Davidson, R.~Sukthankar, and A.~Gupta,
``Robust Adversarial Reinforcement Learning,''
in \textit{Proceedings of the 34th International Conference on Machine Learning (ICML)},
PMLR, vol.~70, pp.~2817--2826, 2017.
\url{https://proceedings.mlr.press/v70/pinto17a/pinto17a.pdf}.

\bibitem{tessler2019actionrobust}
C.~Tessler, Y.~Efroni, and S.~Mannor,
``Action Robust Reinforcement Learning and Applications in Continuous Control,''
in \textit{Proceedings of the 36th International Conference on Machine Learning (ICML)},
PMLR, vol.~97, 2019.
\url{https://proceedings.mlr.press/v97/tessler19a/tessler19a.pdf}.

\bibitem{kristiansson2025warehouse}
L.~Kristiansson and F.~Winkelmann,
``Comparative Analysis of A* and Q-Learning Algorithms for Robot Path Planning in Dynamic Warehouse Environments,''
Bachelor's thesis, KTH Royal Institute of Technology, 2025.
\url{https://www.diva-portal.org/smash/get/diva2:1985734/FULLTEXT01.pdf}.

\bibitem{paczolay2021pursuitevasion}
G.~Paczolay and I.~Harmati,
``A Simplified Pursuit-evasion Game with Reinforcement Learning,''
\textit{Periodica Polytechnica Electrical Engineering and Computer Science},
vol.~65, no.~2, pp.~160--166, 2021.
doi:10.3311/PPee.16540.
\url{https://pp.bme.hu/eecs/article/download/16540/9023/95223}.

\bibitem{watkins1992qlearning}
C.~J.~C.~H.~Watkins and P.~Dayan,
``Q-learning,''
\textit{Machine Learning},
vol.~8, no.~3--4, pp.~279--292, 1992.
doi:10.1007/BF00992698.

\end{thebibliography}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The acknowledgments section is defined using the "acks" environment
%%% (rather than an unnumbered section). The use of this environment 
%%% ensures the proper identification of the section in the article 
%%% metadata as well as the consistent spelling of the heading.

% \begin{acks}
% If you wish to include any acknowledgments in your paper (e.g., to 
% people or funding agencies), please do so using the `\texttt{acks}' 
% environment. Note that the text of your acknowledgments will be omitted
% if you compile your document with the `\texttt{anonymous}' option.
% \end{acks}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The next two lines define, first, the bibliography style to be 
%%% applied, and, second, the bibliography file to be used.

% \bibliographystyle{ACM-Reference-Format} 
% \bibliography{sample}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\end{document}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%