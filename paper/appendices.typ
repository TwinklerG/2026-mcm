#let first-line-indent = 20pt
#let appendices(body) = {
  // show Appendices
  body
  pagebreak()

  pad(
    left: -first-line-indent,
  )[
    #heading("Memorandum", numbering: none)

    #show table.cell: set text(size: 12pt)
    #table(
      columns: (auto, 1fr),
      stroke: none,
      inset: 3pt,
      [*To:*], [Executive Producers, Dancing with the Stars],
      [*From:*], [MCM Team 2601980],
      [*Date:*], [February 2, 2026],
      [*Subject:*], [Recommendations for Voting System Optimization],
    )

    #line(length: 100%, stroke: 0.5pt)
    #v(0.5em)

    This memorandum summarizes our findings from a comprehensive analysis of 34 seasons of DWTS competition data, encompassing 1,411 contestant-weeks and 261 elimination events. Based on this analysis, we recommend adopting the Progressive Technical Fairness System (PTFS) to improve both competitive fairness and audience engagement.

    Our investigation revealed several limitations in the current voting system. The Judges' Save mechanism introduced in Season 28 has not fulfilled its intended purpose of protecting technically superior dancers. In 90.5% of cases where judges exercised this option, they saved the contestant with higher estimated fan support rather than the one with better technical performance ($p = 0.56$). We also identified systematic biases in how different contestant characteristics are evaluated: athletes, for instance, receive lower scores from judges ($beta = -0.129$) but higher fan votes ($beta = +0.052$), effects in opposite directions that the current system cannot reconcile. Furthermore, professional partners explain 2.3 times more variance in fan voting than in judge scoring, indicating that popularity outcomes depend more heavily on partner assignment than on actual dance quality.

    To address these issues, we developed PTFS, which dynamically adjusts the balance between judge scores and fan votes across the competition. The system begins each season with a 45% judge weight (55% fan weight) and increases linearly to 80% judge weight (20% fan weight) by the finale. The combined score is computed as $S_(i,t) = w_J(t) times "Judge"% + (1 - w_J(t)) times "Fan"%$, where $w_J(t)$ follows the linear progression described above.

    When evaluated against the current Percentage-based system across all 34 seasons, PTFS demonstrates meaningful improvements. Kendall's $tau$, which measures the correlation between technical rankings and final placements, increases from 0.727 to 0.749—a statistically significant improvement confirmed by paired $t$-test ($p = 0.031$). The Close Call Rate rises from 66.1% to 79.1%, meaning nearly four out of five eliminations under PTFS are decided by margins of less than 2%, creating substantially more suspense for viewers. Consistency also improves, with 97.1% of seasons achieving $tau > 0.5$ under PTFS compared to 94.1% under the current system.

    We recommend adopting PTFS for several reasons. The improvement is data-driven, having been validated on 34 seasons of historical data, and statistically significant rather than attributable to random variation. Implementation requires only a formula change in score computation with no modifications to voting infrastructure. The principle that "judge scores matter more as we approach the finale" is intuitive and can be communicated transparently to audiences. Most importantly, PTFS serves all stakeholders: technically skilled dancers receive fairer treatment, casual fans enjoy more suspenseful eliminations, and producers benefit from reduced controversy risk and enhanced engagement metrics.

    If the Judges' Save mechanism is retained, we suggest adding explicit criteria requiring judges to evaluate technical merit specifically, as our analysis shows the current implementation provides no meaningful technical protection.

    We are available to discuss implementation details and provide additional analysis as needed.

    #v(0.5em)
    #align(right)[
      _Respectfully submitted,_\
      _MCM Team 2601980_
    ]

    // #v(12pt)
    // *To:* Executive Producers, Dancing with the Stars
    // #v(6pt)
    // *From:* Team 2601980
    // #v(6pt)
    // *Subject:* Recommendations for Vote Aggregation and Competition Fairness
    // #v(6pt)
    // *Date:* February 2, 2026
    // #v(16pt)
    // We have completed a mathematical analysis of 34 seasons of DWTS data to address concerns about competition fairness and the relative influence of judges versus fan votes. This memorandum summarizes our key findings and recommendations.
    // #v(8pt)
    // *Key Findings*
    // #v(4pt)
    // Using a Bayesian inference framework, we recovered latent fan-vote distributions from observed elimination outcomes. The model achieves 95.02% elimination accuracy and Kendall's τ of 0.99 for final rankings. A sensitivity analysis shows that posterior distributions are driven predominantly by elimination constraints rather than prior assumptions (MAD ≈ 0.004, KL ≈ 0.001), supporting the reliability of our inferences.
    // #v(6pt)
    // Our simulation-based comparison of rank-based and percentage-based aggregation methods reveals structural differences in stability and fan influence. The dual-track mixed-effects analysis shows that judges and fans weight celebrity characteristics differently—e.g., athletes receive lower technical scores but higher fan support—and that professional partners explain more variation in fan votes (ICC ≈ 16.4%) than in judge scores (ICC ≈ 9.5%).
    // #v(8pt)
    // *Primary Recommendation: Adopt the Progressive Technical Fairness System (PTFS)*
    // #v(4pt)
    // We propose a new weekly scoring system that progressively shifts weight from fan votes (55%) in early weeks to judges' scores (80%) by the finale. PTFS achieves statistically significant improvement in technical-outcome correlation (p = 0.031) over the current percentage-based system, with higher close-call rates (79.1% vs 66.1%), suggesting more engaging eliminations. Implementation requires only a formula change—no new voting infrastructure.
    // #v(6pt)
    // *Supporting Recommendations*
    // - If retaining the current system, enhance the Judges' Save rule with explicit criteria for technical merit; our analysis indicates the current implementation aligns heavily with fan preferences.
    // - Consider adopting the percentage-based method over the rank-based method where stability is a priority, based on our posterior simulation results.
    // #v(8pt)
    // We are available to discuss these findings and to provide technical support for implementation.
    // #v(8pt)
    // Sincerely,
    // #v(4pt)
    // Team 2601980
  ]
    
  pagebreak()
  
  pad(
    left: -first-line-indent,
  )[
    #heading("Report on Use of AI Tools", numbering: none)
    #set heading(outlined: false)
    
    The AI tools we employed served auxiliary functions similar to spell-checkers or advanced documentation search engines. They assisted with code auto-completion for standard programming patterns, technical documentation queries for library APIs, and language polishing to improve clarity and readability. All outputs were reviewed, validated, and often substantially revised by our team.
    
    == 1. GitHub Copilot
    
    *Purpose:* Code auto-completion during implementation of statistical models and visualization scripts.
    
    *Usage Pattern:* Copilot provided inline suggestions for import statements for scientific libraries (NumPy, Pandas, Polars, Matplotlib, SciPy, Statsmodels), function signatures and parameter names following Python conventions, standard code patterns (e.g., file I/O, data validation, error handling), and docstring templates in NumPy style format.
    
    == 2. Claude
    
    *Purpose:* Technical documentation queries and API usage clarification.
    
    *Example Query:*
    ```
    Query: "What is the Polars syntax for creating a conditional column (if-else logic) based on another column's value?"
    
    Output: "In Polars, you use the `when().then().otherwise()` expression chain inside a context like `with_columns`. For example: `df.with_columns(pl.when(pl.col('score') >= 60).then(pl.lit('Pass')).otherwise(pl.lit('Fail')).alias('status'))`. This allows for vectorized conditional logic without using slow Python loops..."
    ```
    
    == 3. Gemini
    
    *Purpose:* Summarization of relevant research papers and technical articles.
    
    *Usage Pattern:* We used Gemini to generate concise summaries of lengthy research papers on Bayesian inference and vote aggregation methods. This helped us quickly identify key methodologies and results relevant to our analysis.
    
    *Example Query:*
    ```
    Query: "Summarize the main findings of the paper 'Monte Carlo Sampling Methods Using Markov Chains and Their Applications'."
    
    Output: "The paper "Monte Carlo Sampling Methods Using Markov Chains and Their Applications" by W.K. Hastings (1970) is a seminal work that generalized the earlier Metropolis algorithm. Its main finding is the introduction of a specific acceptance probability—now known as the Metropolis-Hastings criterion—which allows for the use of non-symmetric proposal distributions while ensuring the Markov chain still converges to the desired target distribution. This generalization significantly expanded the applicability of MCMC methods to a wider class of problems, allowing for..."
    ```
    
    == 4. ChatGPT
    
    *Purpose:* Language polishing and clarity improvement for the written report.
    
    *Usage Pattern:* We used ChatGPT to refine sentence structures and improve readability of technical explanations. For example rephrasing complex mathematical descriptions for clarity, improving transition sentences between sections and checking grammar and ensuring consistent terminology.
  ]
}