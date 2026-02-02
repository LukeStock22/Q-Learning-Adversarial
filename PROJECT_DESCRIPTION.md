```latex
\begin{document}

\begin{center}
\Large\bfseries Q-Learning in Adversarial Environments \\[0.5em]
Jan 2026
\end{center}

\section{Project Overview}

\subsection*{Background and Motivation}
\textbf{Motivation:} Modern logistics and supply chain management rely heavily on automation. Companies like Amazon utilize thousands of Automated Guided Vehicles (AGVs) in fulfillment centers. However, real-world warehouses are often chaotic. Unexpected events—such as liquid spills, misplaced obstacles, or malfunctioning robots—can disrupt operations.

\textbf{Problem Statement:} Standard path-planning algorithms often fail when the environment changes dynamically. In this project, we aim to model a warehouse floor as a grid-based environment and investigate how Reinforcement Learning (specifically Q-Learning) can be used to train a delivery agent to navigate efficiently while avoiding various types of adversaries.

\section{Existing Literature}

\subsection*{Robust and Adversarial Reinforcement Learning}
A major challenge in reinforcement learning is that policies trained in one setting often perform poorly when the environment changes (different obstructions, dynamics, or disturbances). This can cause a gap between simulated and real-world performance. Pinto et al.~propose \emph{Robust Adversarial Reinforcement Learning (RARL)} which frames robustness as a two-player, zero-sum minimax problem in which an adversary learns to apply destabilizing disturbances while the agent learns to succeed despite them \cite{pinto2017rarl}. Their results in continuous-control benchmarks demonstrate that explicitly training against a learned adversary can improve robustness and sometimes even performance in settings not trained on.

Tessler et al.~ also explore robustness with action uncertainty, where the agent's chosen action may be replaced by an adversarial alternative with some probability, or perturbed in continuous action spaces \cite{tessler2019actionrobust}. They relate these models to real-world robotics uncertainty (e.g., sudden pushes or actuator disturbances) and provide algorithms and evaluations in continuous-control domains.

\subsection*{Reinforcement Learning for Navigation in Dynamic Warehouses}
Warehouse navigation and path planning under changing conditions has also been studied from an applied perspective. Kristiansson and Winkelmann compare A* planning to Q-learning-based approaches for robot path planning in dynamic warehouse environments, evaluating performance primarily using throughput (packages delivered per robot over time) and showing that RL approaches can be more adaptable in highly dynamic scenarios (though it may be prone to overfitting) \cite{kristiansson2025warehouse}. This work supports the practical motivation for learning-based policies when obstacles and traffic patterns vary during operation.  

\subsection*{Multi-Agent and Moving-Opponent Settings}
Dynamic adversaries can also be viewed through a multi-agent lens. Paczolay et al.~studies evasion with collision avoidance and propose a method combining Minimax-Q and Nash-Q to better account for opponents' actions in multi-agent settings \cite{paczolay2021pursuitevasion}. While the domain differs from warehouse logistics, it is a good example of modeling moving opponents and reasoning about adversarial behavior in grid-like environments.

\subsection*{Our Differentiation / Gap}
Prior work motivates our project but there are gaps and unique combinations of features that we will target in our project.The most similar paper explored grid-world warehouses with multiple agents and obstacles \cite{kristiansson2025warehouse} but the obstacles weren't true strategic adversaries. They were just unpredictable/dynamic obstacles. In contrast, other papers have explored robust adversarial learning in a continuous control robotics environment (MuJoCo) like \cite{pinto2017rarl} and \cite{tessler2019actionrobust}, or framed as a capture scenario \cite{paczolay2021pursuitevasion}. But these did not model warehouse scenarios or the specific obstructions that might occur in such a scenario. As a point of differentiation, we will explicitly compare training under (i) non-adversarial dynamic obstacles and (ii) a constrained strategic adversary, and evaluate which training style produces policies that generalize better to unseen conditions (e.g., new layouts, different obstacle motion patterns, or different disruption intensities). Specifically, we aim to answer whether adversarially-trained warehouse policies are more robust than policies trained only under stochastic dynamics.

\section{Project Description}

\textbf{The Environment:} We will model the warehouse as an $N \times N$ Gridworld. The agent is a delivery robot tasked with retrieving a package and delivering it to a designated zone efficiently.

\textbf{The Adversarial Models:} To rigorously test robustness, we differentiate between \textit{stochastic disturbances} and \textit{strategic adversaries}:
\begin{enumerate}
    \item \textbf{Baseline: Stochastic Disturbances (``Nature'')}
    \begin{itemize}
        \item \textbf{Static Spills:} Randomly placed obstacles simulating liquid spills or dropped boxes.
        \item \textbf{Random Forklifts:} Non-adversarial agents that move randomly, representing background traffic that requires collision avoidance.
    \end{itemize}
    
    \item \textbf{Differentiation: Strategic Adversaries (``Attacker'')}
    \begin{itemize}
        \item \textbf{Congestion Adversary:} A constrained adversary that can block specific edges or aisles (on a budget) to maximize the agent's path length, simulating worst-case congestion.
        \item \textbf{Jammer Robot:} An adversarial agent that actively moves to block the delivery robot's path, creating a dynamic pursuit-evasion scenario.
    \end{itemize}
\end{enumerate}

\textbf{Methodology:} 
We will implement \textbf{Tabular Q-Learning} to train two sets of policies:
\begin{enumerate}
    \item \textbf{Policy A (Stochastic Training):} Trained only against random spills and forklifts.
    \item \textbf{Policy B (Adversarial Training):} Trained against the strategic Jammer/Congestion adversary.
\end{enumerate}
Our core experiment is to evaluate whether Policy B generalizes better to \textit{unseen} environment configurations compared to Policy A, thereby proving that adversarial training improves robustness.
\subsection*{Our Position and Differences}

\section{Potential Expansions}
\begin{enumerate}
    \item Vary hyperparameters e.g. test different grid sizes, congestion levels, shelving
    \item Test other realistic real-world disturbances that add another dimension to adversarial behavior
    \item Explore other domains e.g. football running play or traversing a real-world style map with various roads that support different speeds or pac-man
\end{enumerate}


\section{Plan}
\textbf{Week 1--2 (Jan 26--Feb 9):}
Establish the project foundation and define the core setup.
\begin{itemize}
  \item Clarify the problem setting, success criteria, and key assumptions.
  \item Build an initial Gridworld-style environment and basic task loop (e.g., pickup $\rightarrow$ delivery).
  \item Implement an initial version of tabular Q-learning and confirm end-to-end training runs.
  \item Set up basic logging/visualization for performance over training.
\end{itemize}

\noindent\textbf{Week 3--4 (Feb 10--Feb 23) \emph{Progress report due Feb 18}:}
Produce early results and refine the experimental direction.
\begin{itemize}
  \item Add one or more forms of environment variability (e.g., static and/or moving disruptions).
  \item Run preliminary experiments to verify the setup behaves as expected.
  \item Implement at least one simple baseline for comparison (e.g., a classical planning or heuristic approach).
  \item Prepare the progress report with initial findings and an updated plan.
\end{itemize}

\noindent\textbf{Week 5--6 (Feb 24--Mar 9):}
Expand the environment model and begin robustness-focused evaluation.
\begin{itemize}
  \item Introduce additional disruption models and tune their difficulty/parameters.
  \item Develop an evaluation protocol that tests performance under multiple conditions.
  \item Continue iterating on training stability and experiment organization.
\end{itemize}

\noindent\textbf{Week 7--8 (Mar 10--Mar 23):}
Main experimentation phase.
\begin{itemize}
  \item Run a broader set of experiments comparing training conditions and environment settings.
  \item Explore generalization to ``unseen'' conditions (e.g., different layouts or disruption patterns).
  \item Conduct ablations or controlled comparisons to isolate which factors matter most.
\end{itemize}

\noindent\textbf{Week 9--10 (Mar 24--Apr 6):}
Analysis, visualization, and optional extensions.
\begin{itemize}
  \item Consolidate results into a small set of core plots/tables.
  \item Perform qualitative analysis of representative successes/failures.
  \item If time allows, add a limited extension (e.g., constraints, partial observability, or additional baselines) and evaluate its impact.
\end{itemize}

\noindent\textbf{Week 11--12 (Apr 7--Apr 20) \emph{Final presentation Apr 22}:}
Finalize deliverables and presentation.
\begin{itemize}
  \item Write the final report and ensure the narrative matches experimental evidence.
  \item Prepare presentation slides and (optionally) a lightweight demonstration/visualization.
  \item Clean up code/experiments for basic reproducibility and finalize figures.
\end{itemize}



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

\end{thebibliography}

\end{document}
```
