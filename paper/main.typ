#import "@preview/showybox:2.0.4": showybox

#import "template.typ": project
#import "appendices.typ": appendices
#show: appendices.with()

// definition
#let argmin = math.op("arg min", limits: true)
#let argmax = math.op("arg max", limits: true)

// TODO Summary一定要把结果写上，展示好的数据
// 不要在摘要里引用
#show: project.with(
  title: "The Mathematics of Stardom: Insights into Fan Vote Inference and Aggregative Rules",
  abstract: [
    // 《与星共舞》（DWTS）综合评委评分和观众投票来决定每周的淘汰名单和最终排名，各季采用基于排名或百分比的规则。这种混合系统旨在平衡技术水平和人气，但也引发了一些争议，例如低分选手凭借粉丝支持晋级，这引发了人们对公平性、偏见和稳定性的质疑。我们开发了一个统一的数学框架来估算未观测到的粉丝投票，评估汇总方法，量化影响因素，并提出一个改进的系统。
    Dancing with the Stars (DWTS) aggregates judges' scores and audience votes to determine weekly eliminations and final rankings, employing rank-based or percentage-based rules across seasons. This hybrid system aims to balance technical merit with popularity but has sparked controversies, such as low-scoring contestants advancing via fan support, raising questions of fairness, bias, and stability. We develop a unified mathematical framework to estimate unobserved fan votes, evaluate aggregation methods, quantify influencing factors, and propose an improved system.

    // 首先，我们将粉丝投票份额建模为一个*约束贝叶斯逆问题*，通过马尔可夫链蒙特卡罗(MCMC)采样，在基于淘汰的不等式约束和时间平滑先验下推断后验分布。该方法实现了*93.49%的淘汰准确率*，*预测排名与观测排名之间的肯德尔τ系数为0.994*，*成功识别出*全部34季的冠军，*100%的情况下*恢复了前三名的排名顺序。
    First, we model fan vote shares as a *constrained Bayesian inverse problem*, inferring posterior distributions via MCMC sampling under elimination-based inequality constraints and temporal smoothness priors. The approach achieves *93.49% elimination accuracy*, *Kendall's $tau$ = 0.994* for predicted vs. observed rankings, identifies the champion in *all 34 seasons*, and recovers the top-3 ordering in *100% of cases*.

    // 其次，利用后验粉丝投票样本，我们模拟了基于排名和基于百分比规则下的结果，发现两种规则在*10.2%*的淘汰周中存在分歧。在分歧周中，基于百分比的方法表现出更大的粉丝偏好。对于有争议的选手（例如，杰里·赖斯、比利·雷·赛勒斯、布里斯托尔·佩林、鲍比·博恩斯），两种方法产生了不同的结果轨迹，基于百分比的规则更有利于粉丝驱动的选手晋级。裁判拯救规则（第 28 至 34 赛季）在 90.5% 的情况下与更高的粉丝份额一致。
    Second, using posterior fan-vote samples, we simulate outcomes under rank- and percentage-based rules, finding they disagree in *10.2%* of elimination weeks. In disagreement weeks, the percentage-based method exhibits greater fan bias. For controversial contestants (e.g., Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones), the methods yield divergent trajectories, with percentage-based rules favoring fan-driven survival. The Judges' Save rule (Seasons 28–34) aligns with higher fan shares in *90.5%* of cases, failing to prioritize technical merit.

    // 第三，双轨分层混合效应模型将评委评分与粉丝投票区分开来，结果显示，专业搭档仅能解释评委评分9.5%的方差，却能解释粉丝投票21.8%的方差——其对人气的影响比对技术评价的影响强2.3倍。名人特质也呈现出类似的差异：运动员的评委评分较低，但粉丝支持率较高；年龄对技术表现的负面影响大于人气。这些模式证实，搭档分配和关键名人特质对粉丝支持的影响远大于对评委评分的影响，验证了该模型对不同评价机制的分离。
    Third, a dual-track hierarchical mixed-effects model disentangles judges' scores from fan votes, showing professional partners explain only *9.5%* of variance in judges' scores but *21.8%* in fan votes — a *2.3-fold stronger effect* on popularity than on technical evaluation. Celebrity traits diverge similarly: athletes receive lower judges' scores yet higher fan support, while age impacts technical performance more negatively than popularity. These patterns confirm that both partner assignment and key celebrity traits exert a *substantially stronger influence on fan support than on judge-assessed outcomes*.

    // 最后，我们提出了*渐进式技术公平系统（PTFS）*，该系统逐步提高评委评分权重，从赛季初的*45%*提高到决赛的*80%*，从而平衡了赛季初期的观众悬念和赛季后期的技术优先性。对全部34季的模拟结果表明，PTFS优于基于排名和基于百分比的规则，将肯德尔τ值提高到*0.749*（分别从0.674和0.727提高到0.749*），将接近淘汰率提高到*79.1%*（分别从30.2%和66.1%提高到0.79.1%*），并提供了更强的技术公平性、更高的观众兴奋度和更佳的跨季稳定性——对于未来的《与星共舞》及类似比赛而言，这是一项强有力的升级。
    Finally, we propose the *Progressive Technical Fairness System (PTFS)*, which progressively increases judges' score weight from *45%* early to *80%* in the finale, balancing initial fan suspense with late-season technical priority. Simulations over all 34 seasons demonstrate PTFS outperforms both rank-based and percentage-based rules, improving Kendall's $tau$ to *0.749* (from 0.674 and 0.727 respectively), boosting close-call elimination rates to *79.1%* (from 30.2% and 66.1%), and providing stronger technical fairness, viewer excitement, and cross-season stability — a robust upgrade for future Dancing with the Stars and analogous competitions.

  ],
  keywords: (
    "Bayesian inference",
    "Latent vote estimation",
    "Voting aggregation rules",
    "Fairness analysis",
    "Mixed-effects models",
  ),
  team-number: "2601980",
  problem-chosen: "C",
  year: "2026",
  bibliography-file: "refs.bib",
)

#set table(stroke: none)
#show table.cell: set text(size: 11pt)
#show figure.caption: set text(size: 11pt)

= Introduction

== Background
// 《与星共舞》（Dancing with the Stars, DWTS）是一档以竞赛形式呈现的娱乐节目，其比赛结果由专业评委打分与观众粉丝投票共同决定。该评分机制旨在在舞蹈技术评价与公众审美偏好之间取得平衡，使比赛既具专业性，又具广泛的观众参与度。

// 在节目长期播出的过程中，评委评分与粉丝投票之间的差异逐渐显现，并在部分赛季中引发了较大争议。一些明星选手在评委评分持续偏低的情况下，仍然凭借观众支持获得较高名次，甚至最终获胜。这类现象引发了观众与媒体对比赛公平性及赛制合理性的持续关注。

// 在此背景下，基于历史赛季数据构建合理的数学模型，不仅有助于理解 DWTS 中评委与观众力量的相对作用，也能够为类似娱乐竞技节目的赛制设计提供可量化的参考依据。

//这里还是不要简略了，可以放一张图像

Dancing with the Stars (DWTS) is a competitive entertainment television program in which the final results are determined by a combination of judges' scores and audience votes. This scoring system is designed to balance professional evaluations of dance performance with public aesthetic preferences, thereby ensuring both technical credibility and broad audience participation.

Throughout the show's long broadcast history, differences between judges' scores and audience voting outcomes have gradually become more apparent and have led to controversy in several seasons. In particular, some celebrity contestants received consistently low scores from the judges but still achieved high rankings, or even won the competition, due to strong fan support. Such results have raised concerns among viewers and the media regarding the fairness of the competition and the rationality of the voting mechanism.

In this context, developing a mathematical model based on historical season data is meaningful. Such a model can be used to analyze the relative influence of judges' scores and audience votes, and it also provides a quantitative reference for the design of voting systems in similar entertainment competitions.

#figure(
  image("img/DWTS_New_Logo.png", width: 70%),
  caption: [
    Dancing with the Stars.
  ],
) <fig:logo>
// Dancing with the Stars (DWTS) is a televised dance competition where celebrities, paired with professional dancers, are evaluated weekly by expert judges and audience votes. The hybrid scoring mechanism—combining technical assessment with popular preference—has generated recurring controversies: contestants with persistently low judge scores have advanced or won through fan support alone (e.g., Bobby Bones in Season 27). These outcomes raise fundamental questions about fairness, stability, and the appropriate balance between expert evaluation and public participation in entertainment competitions.

== Restatement of the Problem
// Task 1 ~ 4 分别重述
Considering the background information and restricted conditions identified in the problem statement, we need to solve the following problems:
// 建立模型推断每位选手观众投票的分布；评估估计与淘汰结果的一致性，并给出可随选手与周次变化的不确定性度量。
- Task 1: Develop a model to estimate the distribution of fan votes, evaluate the consistency and uncertainty of the estimates, and visualize the results.

// TODO 基于观众投票估计结果与其余比赛数据，
// 对比各赛季中“按排名综合”与“按百分比综合”两种投票合成方式的比赛结果，分析其对观众投票影响程度的偏向性；
// 进一步针对争议选手，研究不同合成方式以及“由评委在垫底两名中选定淘汰者”规则对比赛结果的影响，
// 并据此为未来赛季推荐更合理的投票合成方式及相关规则设置。
- Task 2: Using the estimated fan votes and the remaining competition data:

  - *First*, compare the outcomes of the rank-based and percentage-based vote aggregation methods across seasons and analyze the degree to which each method favors fan influence.
  - *Second*, examine the effects of different aggregation methods and the rule allowing judges to select the eliminated contestant from the bottom two on the outcomes of controversial contestants.
  - *Third*, based on the analysis, recommend an appropriate vote aggregation method and determine whether the judges' selection rule should be retained for future seasons.

// 基于含观众投票估计的数据，建模分析职业舞伴与明星特征（年龄、行业等）对比赛成绩的影响，并比较其对评委分数与观众投票的影响是否一致。
- Task 3: Develop a model to analyze the impact of professional dancers and celebrity characteristics (such as age and industry) on competition performance, and compare whether these factors affect judges' scores and fan votes in the same way.

// 提出一套新的、每周结合观众投票与评委分数的赛制，论证其更“公平”或在其他维度更优，并说明为何值得被节目制作方采纳。
- Task 4: Propose a new weekly competition system that combines fan votes and judges' scores, justify why it is more fair or superior in other aspects, and explain why it should be adopted by the show producers.

== Our work
// 流程图 + 流程概述
#figure(
  image("img/workflow.jpg", width: 100%),
  caption: [
    The workflow of our work.
  ],
) <fig:workflow>

// @fig:workflow presents the high-level workflow of our methodology.
// 最好不要用(fig)这种形式
Our approach systematically addresses the four tasks through interconnected analytical frameworks. @fig:workflow presents the high-level workflow of our methodology. First, in the *Bayesian Inference Framework*@gelman2013bayesian, we estimate latent fan-vote distributions from elimination outcomes using MCMC sampling@Hastings1970MonteCS with data-driven priors. Second, in *Simulation-Based Outcome Analysis*, we compare rank-based and percentage-based aggregation methods across seasons to evaluate their impact on elimination stability and fairness. Third, in the *Dual-Track Mixed-Effects Model*@Gelman_Hill_2006, we separately model judges' scores and fan votes to quantify how professional partners and celebrity characteristics differentially affect technical evaluation versus popularity outcomes. Finally, the *Progressive Technical Fairness System* (PTFS) proposes a dynamic weighting scheme derived from social choice theory@arrow1951social that progressively adjusts the balance between judges' influence and fan votes throughout the season.

= Assumptions and Justification
// 假设与合理性说明

*Fan vote shares are latent and identified by elimination constraints.*
Fan votes are unobserved. The weekly elimination order is mechanically determined by the official aggregation rule applied to judges' scores and fan votes. Therefore, observed elimination outcomes impose inequality constraints on the latent fan vote shares, which are exploited for identification and posterior inference in Task 1.
// 粉丝投票份额为潜变量，由淘汰约束识别。粉丝投票不可直接观测；每周淘汰结果由官方合成规则机械地作用于评委分与粉丝投票而生成，因此淘汰顺序对潜在粉丝投票份额施加不等式约束。我们在 Task 1 中正是利用这些约束对粉丝投票后验分布进行识别与推断。

*Fan vote shares evolve smoothly across adjacent weeks.*
A contestant's fan vote share is assumed to change gradually rather than exhibit abrupt structural jumps from one week to the next within a season. We encode this assumption through a temporal smoothing prior in Task 1, which regularizes the posterior distribution and renders the inverse problem well-posed. While this assumption may not hold during rare exogenous shocks, it provides a necessary regularization for identifying weekly vote shares across the vast majority of the competition's timeline.
// 粉丝投票份额在相邻周之间平滑演化。我们假设同一赛季内，选手的粉丝投票份额随时间逐步变化，而不会在相邻周发生无原因的剧烈跳变。该假设通过 Task 1 中的时间平滑先验 进行刻画，用以正则化后验并保证逆问题的适定性。

*Judges' scores and fan votes represent distinct outcome mechanisms.*
Judges' scores primarily reflect technical evaluation, whereas fan votes capture popularity and viewer preference. Consequently, we model the two outcomes using separate tracks in Task 3 and examine how professional partners and celebrity attributes differentially affect each mechanism.
// 评委评分与粉丝投票对应不同的结果生成机制。评委评分主要反映技术评价，而粉丝投票体现人气与观众偏好。因此在 Task 3 中我们对二者进行分轨建模，并比较职业舞伴与明星个体特征对两种机制的差异化影响。


// 【符号表已注释：符号在使用时定义，节省页面空间】
= Notations
// 符号说明

#figure(
  table(
    columns: (1fr, 6fr),
    align: (center, left),
    inset: 4pt,
    table.hline(),
    [*Symbol*], [*Description*],
    table.hline(),
    [$t$], [Competition week index],
    [$i$], [Contestant (celebrity) index],
    [$p(i)$], [Professional partner of contestant $i$],
    [$C_t$], [Set of contestants still competing in week $t$],
    [$J_(i,t)$], [Total judges' score for contestant $i$ in week $t$],
    [$f_(i,t)$], [Fan vote share (proportion) for contestant $i$ in week $t$; latent],
    [$bold(f)_t$], [Vector of fan shares in week $t$],
    [$E_t$], [Set of contestants eliminated in week $t$ (observed)],
    [$mu_i$, $sigma_"prior"$], [Prior mean and standard deviation for fan share],
    [$S$], [Number of MCMC posterior samples],
    [$T$], [Total number of weeks in a season],
    [$Y_(i,t)^J$, $Y_(i,t)^F$], [Judges' score and fan-vote outcome for celebrity $i$ in week $t$],
    [$bold(X)_i$], [Celebrity covariate vector (e.g., age, industry)],
    [$beta_0^J$, $bold(beta)^J$, $beta_w^J$], [Fixed intercept, covariate effects, and week trend in judges' model],
    [$beta_0^F$, $bold(beta)^F$, $gamma$],
    [Fixed intercept, covariate effects, and judges'-score effect in fan-vote model],
    [$u_(p(i))^J$, $u_(p(i))^F$], [Partner random effect (judges' track and fan-vote track)],
    [$sigma_u^2$, $sigma_epsilon^2$], [Partner random-effect variance and residual variance],
    // [ICC], [Intraclass correlation coefficient; fraction of variance explained by partner],
    // [IDI], [Impact divergence index; standardized difference of covariate effect between tracks],
    [$hat(u)_(p(i))$], [BLUP of partner random effect],
    
    table.hline(),
  ),
  caption: none,
)<tab:notations>

= Data Preprocessing
== Data Analysis
// 数据分析
The dataset comprises weekly performance records, individual judge scores, season outcomes, and professional–celebrity partnership information. Over the course of *Dancing with the Stars*, certain institutional features—such as the size of the judging panel, elimination rules, and score aggregation procedures—have varied across seasons. While the core data structure remains stable, these design changes introduce systematic differences that must be accounted for prior to modeling. Accordingly, data preprocessing emphasizes institutional sources of missingness and variation that are inherent to the competition format.

// 数据集包含逐周表现记录、评委单项评分、赛季结果以及职业舞伴—明星搭档信息。在《与星共舞》的长期播出过程中，部分制度性要素（如评委人数、淘汰规则及评分合并方式）在不同赛季间有所调整。尽管数据的基本结构保持稳定，这些赛制变化仍引入了需要在建模前加以处理的系统性差异。因此，数据预处理侧重于识别并处理源自赛制安排的缺失与变异。

Missing judge scores arise from clearly defined structural mechanisms rather than random loss. As shown in @tab:missing-sources, a portion of score fields is unused in seasons or weeks with fewer judges, while additional missing values occur mechanically after contestant elimination, when no further weekly scores are generated.
// 评委评分中的缺失值主要源于明确的结构性原因，而非随机缺失。如表 @tab:missing-sources 所示，部分评分字段在评委人数较少的赛季或周次中未被使用；此外，选手被淘汰后不再产生后续周次评分，从而形成制度性缺失。

#figure(
  table(
    columns: (1.4fr, 4fr, 1fr),
    align: (left, left, center),
    table.hline(),
    [*Source*], [*Description*], [*Count*],
    table.hline(),
    [Judge slot not used], [Fewer judges in certain seasons or weeks], [2,499],
    [Post-elimination], [No scores recorded after contestant elimination], [6,904],
    [], [Total score cells], [18,524],
    table.hline(),
  ),
  caption: [Sources of missing values in judge score records.],
)<tab:missing-sources>

In addition to missingness, the elimination mechanism itself varies across seasons. While most weeks involve a single elimination, some feature double eliminations, and a small number involve three contestants eliminated simultaneously. @fig:eliminations-per-week summarizes, for each competition week, the number of seasons exhibiting different elimination counts. This variation implies that elimination outcomes cannot be inferred from score rankings alone without conditioning on the contemporaneous rules.
// 除缺失问题外，淘汰机制在不同赛季之间亦存在差异。多数周次淘汰一名选手，但部分周次淘汰两名，极少数周次同时淘汰三名选手。图 @fig:eliminations-per-week 按周统计了各赛季中不同淘汰人数的分布情况。这一制度差异意味着，淘汰结果不能仅由评分排序直接推断，而必须结合当周赛制规则加以理解。

#figure(
  image("img/eliminations_per_week_stacked.png", width: 90%),
  caption: [Distribution of elimination counts per week across seasons.],
)<fig:eliminations-per-week>

Observed score values at the boundaries of the empirical range are retained and interpreted within their institutional context. Zero scores correspond to post-elimination records, while values exceeding standard score limits arise from competition-specific bonus rules. Exceptional events such as contestant withdrawals are encoded through indicator variables to preserve the structural integrity of the dataset.
// 对于位于经验分布边界的评分值，本文结合其制度背景进行解释并予以保留。其中，评分为零的记录对应选手被淘汰后的观测值，而超过常规评分上限的数值来源于赛制中的奖励分规则。退赛等特殊事件通过指示变量加以标识，以保持数据结构的完整性。


== Data Transformation
// 在完成数据清洗后，本文对原始评分数据的结构进行了重组，以适应时间序列分析的需要。原始评分数据以宽表形式存储，不同周次及评委评分分别对应不同字段，不利于时间维度上的统计建模。为此，本文将其转换为长表结构，使每一条记录对应“选手—周次—评委—得分”的组合。
Following data cleaning, the raw score data were restructured to support time-series analysis. The original wide-format table, in which scores from different weeks and judges were stored in separate columns, was unsuitable for temporal modeling. Accordingly, the data were transformed into a long-format structure, where each record represents a *contestant–week–judge–score* combination (@tab:long-format).

#figure(
  table(
    columns: (1.8fr, 4fr),
    align: (left, left),
    table.hline(),
    [*Column*], [*Description*],
    table.hline(),
    [`contestant_id`], [Contestant identifier (e.g. celebrity `name` + `season`)],
    [`season`], [Season number],
    [`week`], [Competition week index],
    [`judge`], [Judge index within the week],
    [`score`], [Judge score for that contestant–week–judge],
    table.hline(),
  ),
  caption: [Long-format score schema: one row per contestant–week–judge combination.],
)<tab:long-format>

// 在长表结构基础上，本文进一步对数据进行逐周汇总处理。具体而言，在剔除选手被淘汰后产生的零分记录后，按“选手—周次”对评分数据进行聚合，计算得到每位选手在每一周的总得分与平均得分。同时，在同一赛季内部，根据周总分对选手进行排序，得到逐周排名信息。
Based on the long-format data, weekly aggregation was performed. After excluding zero-score entries generated after elimination, scores were grouped by *contestant and week* to compute weekly total and average scores. Within each season, contestants were ranked by weekly total score to obtain week-level rankings.

// 为实现不同时间尺度之间的数据统一，本文在逐周汇总数据的基础上构建赛季层面的统计量。通过对选手在整个赛季中的逐周表现进行聚合，计算得到赛季参赛周数、赛季总得分、赛季平均得分及平均排名等指标。随后，将逐周数据、赛季统计结果以及搭档信息表进行合并，形成包含时间信息、表现指标及背景变量的综合分析数据集。
To integrate information across time scales, season-level statistics were constructed from the weekly summaries. These include the number of weeks competed, total season score, average weekly score, and average rank. The weekly data, season-level statistics, and partner information were then merged to form a comprehensive dataset containing temporal, performance, and contextual variables.

== Feature Recombination
// 在完成数据结构转换后，本文进一步从不同时间尺度和信息来源对变量进行重组，以构建用于模型分析的特征集合。
After data transformation, features were recombined from multiple temporal scales: 

(1) *weekly-level* features capturing short-term dynamics (previous score, week-to-week change, deviation from cumulative average); 

(2) *season-level* statistics describing long-term performance (total score, average rank, score volatility); and 

(3) *partner-level* features summarizing professional dancer competitiveness (championships, top-3 finishes, average judge score). Categorical variables such as industry were encoded into model-compatible formats (@tab:features).

#figure(
  table(
    columns: (1fr, 2fr, 2.5fr),
    align: (left, left, left),
    table.hline(),
    [*Scale*], [*Variable*], [*Description*],
    table.hline(),
    [Weekly], [`prev_week_score`], [Previous week total score],
    [], [`score_change`], [Week-to-week score change],
    [], [`deviation_from_avg`], [Deviation from cumulative average],
    table.hline(),
    [Season], [`weeks_competed`], [Number of weeks competed],
    [], [`season_avg_weekly_score`], [Season average weekly score],
    [], [`score_volatility`], [Score standard deviation],
    [], [`avg_rank`], [Average weekly rank],
    table.hline(),
    [Partner], [`championships`], [Partner's championship count],
    [], [`top3_finishes`], [Partner's top-3 finish count],
    [], [`avg_judge_score`], [Partner's historical average score],
    table.hline(),
    [Other], [`industry_code`], [Celebrity industry category],
    [], [`is_controversy_case`], [Binary indicator for controversy],
    table.hline(),
  ),
  caption: [Engineered features across temporal scales.],
)<tab:features>


To diagnose redundancy and multi-collinearity across these feature groups, we also examined Spearman correlation matrices@spearman04 of key variables at the weekly, season, and partner levels. As shown in @fig:heat, the three-level correlation structure shows that most engineered features are moderately correlated rather than collinear, supporting their joint use in subsequent models while flagging a few strongly associated pairs to be interpreted with caution.

#figure(
  image("img/13_correlation_matrices_three_levels.png", width: 100%),
  caption: [Spearman correlation matrices.],
)<fig:heat>

= Task 1: Bayesian Inference Framework for Estimating Fan Vote Distributions

== Model Formulation
The fundamental challenge in analyzing *Dancing with the Stars* (DWTS) voting dynamics is that actual fan votes are never disclosed. We therefore formulate this as a *constrained Bayesian inverse problem* @stuart2010inverse.

Let $C_t = {c_1, c_2, …, c_n}$ denote the set of $n$ contestants competing in week $t$. For each contestant $i$, define $J_i ∈ [0, 100]$ as the total judges' score, $E_t ⊂ C_t$ as the observed eliminated set, and $f_(i,t) ∈ (0, 1)$ as the latent fan vote share satisfying the simplex constraint
$
sum_i f_(i,t) = 1.
$

The fan vote vector
$
bold(f)_t = (f_(1,t), …, f_(n,t))
$
is not directly observable, and elimination outcomes provide only limited ranking information. We therefore seek the posterior distribution
$
P(bold(f)_t | E_t, bold(J)_t)
prop
P(E_t | bold(f)_t, bold(J)_t)
dot.op
P(bold(f)_t | bold(f)_(t-1))
dot.op
P(bold(f)_t),
$
where the three terms correspond to the *Elimination-Consistency likelihood*, *temporal prior*, and *static prior*, respectively.

*The likelihood* encodes elimination rules as hard constraints. DWTS has employed two aggregation schemes: the rank-based method, where
$
"Score"_i = "Rank"(J_i) + "Rank"(f_(i,t)),
$
and the percentage-based method, where
$
"Score"_i = J_i / (sum_j J_j) + f_(i,t).
$

Let $k_t = |E_t|$ denote the number of eliminations in week $t$. For standard weeks (Seasons 1–27), the likelihood is given by the indicator function
$
P(E_t | bold(f)_t, bold(J)_t)
=
bb(I)(text("Bottom")-k_t = E_t).
$

From Season 28 onward, the *Judges' Save* rule allows judges to choose whom to eliminate from the bottom two. Accordingly, the constraint relaxes to requiring that the observed elimination set be contained within this bottom-ranked subset.

*The prior distribution*
$
f_(i,t) ~ cal(N)(mu_i, sigma_"prior"^2),
quad
sigma_"prior" = 0.25,
$
incorporates industry-specific base rates derived from historical survival analysis.

To account for the "longevity effect", where research suggests established media presence drives fan loyalty and para-social interaction @ferchaud2018parasocial, we apply a mild age adjustment factor. This mechanism, designed to peak at age 40 to reflect the prime of established stardom, is modeled as:
$
mu_i <- mu_i dot.op
(0.9 + 0.1 dot.op exp(-("Age"_i - 40)^2 / 30^2)),
$
which peaks at age 40 and captures variation in audience preferences across age groups.

Finally, since fan voting behavior typically evolves smoothly across consecutive weeks, a *temporal smoothing prior*
$
P(bold(f)_t | bold(f)_(t-1))
prop
exp(-lambda sum_i (f_(i,t) - f_(i,t-1))^2),
quad
lambda = 0.5,
$
regularizes week-to-week changes and suppresses implausible abrupt fluctuations in fan voting behavior.

A comparative analysis confirms that while both data-driven and uniform priors were tested, their impact on the core latent vote estimates for surviving contestants is negligible (survivor posterior Mean Absolute Difference < 0.005). Given this robustness, we selected the industry-specific prior not for its marginal impact on elimination accuracy, but for its theoretical value in formally embedding domain knowledge and providing a crucial foundation for the partner-effect analysis in Task 3.

// The fundamental challenge in analyzing DWTS voting dynamics is that actual fan votes are never disclosed. We therefore formulate this as a *constrained Bayesian inverse problem*@stuart2010inverse. 

// Let $C_t = {c_1, dots, c_n}$ denote the set of $n$ contestants competing in week $t$. For each contestant $i$, define $J_i in [0, 100]$ as the total judges' score, $E_t subset C_t$ as the observed eliminated set, and $f_(i,t) in (0, 1)$ as the latent fan vote share satisfying the simplex constraint $sum_i f_(i,t) = 1$. The fan vote vector $bold(f)_t = (f_(1,t), dots, f_(n,t))$ is not directly observable, and elimination outcomes provide only limited ranking information. We therefore seek the posterior distribution

// $
//   P(bold(f)_t | E_t, bold(J)_t)
//   prop
//   P(E_t | bold(f)_t, bold(J)_t)
//   dot.op
//   P(bold(f)_t | bold(f)_(t-1))
//   dot.op
//   P(bold(f)_t)
// $
// where the three terms correspond to the likelihood, temporal prior, and static prior, respectively.

// The likelihood encodes elimination rules as hard constraints. DWTS has employed two aggregation schemes: the rank-based method (Seasons 1–2 and 28–34), where $"Score"_i = "Rank"(J_i) + "Rank"(f_(i,t))$, and the percentage-based method (Seasons 3–27), where $"Score"_i = J_i / (sum_j J_j) + f_(i,t)$. Let $k_t = |E_t|$ denote the number of eliminations in week $t$. For standard weeks (Seasons 1–27), the likelihood is $P(E_t | bold(f)_t, bold(J)_t) = bb(I)("Bottom"-k_t = E_t)$. From Season 28 onward, the Judges' Save rule allows judges to choose whom to eliminate from the bottom two, so the constraint relaxes to requiring $E_t$ be contained in that set.

// The prior distribution $f_(i,t) ~ cal(N)(mu_i, sigma_"prior"^2)$ incorporates industry-specific base rates derived from historical survival analysis ($sigma_"prior" = 0.25$). Research suggests that longevity and established media presence are key drivers of parasocial interaction and fan loyalty@ferchaud2018parasocial. In the context of DWTS, this 'longevity effect' typically culminates in mid-career celebrities. We therefore operationalize this mechanism through a mild age adjustment peaking at 40 (reflecting the prime of established stardom), modeled as: $mu_i <- mu_i dot.op (0.9 + 0.1 dot.op exp(-("Age"_i - 40)^2 / 30^2))$. A temporal smoothing prior $P(bold(f)_t | bold(f)_(t-1)) prop exp(-lambda sum_i (f_(i,t) - f_(i,t-1))^2)$ with $lambda = 0.5$ regularizes week-to-week changes. While both data-driven and uniform priors yield posterior distributions satisfying elimination constraints (survivor posterior MAD $< 0.005$), we adopt industry-specific priors to incorporate domain knowledge and facilitate downstream partner-effect analysis in Task 3.

#figure(
  image("img/Markov-chain-Monte-Carlo-sampling-using-random-walk.png", width: 70%),
  caption: [Markov-chain-Monte-Carlo Sampling.@markovchainfigure],
)<fig:mcmc>

== Posterior Distribution and Metropolis-Hastings Sampling

Combining the static prior and temporal smoothness terms, the posterior log-probability can be written as
// 将静态先验与时间平滑项结合，后验对数概率可写为
$
  log P(bold(f)_t)
  prop
  - 1 / (2 sigma_"prior"^2) sum_i (f_(i,t) - mu_i)^2
  - lambda sum_i (f_(i,t) - f_(i,t-1))^2
$
subject to the simplex constraint and the elimination-consistency condition. If the likelihood constraint is violated, we set
// 且须满足单纯形约束与淘汰一致性条件；若违反似然约束，则令
$
  log P(bold(f)_t) = -infinity
$


Since the posterior has no closed form, we employ adaptive Metropolis-Hastings sampling@gelman2013bayesian. Proposals are made in log-space and renormalized to the simplex:
$
  log(bold(f)^*) = log(bold(f)^((k))) + epsilon, quad epsilon ~ cal(N)(0, sigma_"prop"^2).
$
Acceptance uses $alpha = min(1, P(bold(f)^*) / P(bold(f)^((k))))$. An adaptive mechanism tunes the proposal scale to target an acceptance rate of approximately 0.35; thinning (every 5th draw retained) reduces autocorrelation.
@fig:fan-share-dist shows the distribution of estimated fan vote shares for eight selected seasons, illustrating how inferred shares concentrate at low values when many contestants remain and spread toward higher values as the field narrows.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 10pt,
    image("img/fan_share_histogram_s01.png", width: 100%),
    image("img/fan_share_histogram_s05.png", width: 100%),
    image("img/fan_share_histogram_s10.png", width: 100%),
    image("img/fan_share_histogram_s15.png", width: 100%),
    
    image("img/fan_share_histogram_s20.png", width: 100%),
    image("img/fan_share_histogram_s25.png", width: 100%),
    image("img/fan_share_histogram_s30.png", width: 100%),
    image("img/fan_share_histogram_s34.png", width: 100%),
  ),
  caption: [
    Distribution of estimated fan vote shares.
  ],
) <fig:fan-share-dist>

== Evaluation Metrics

We assess model performance using four complementary metrics. 
// Elimination accuracy measures the proportion of weeks where the predicted bottom-$k_t$ set exactly matches the observed eliminations: $"Acc" = 1 / T sum_(t=1)^T bb(I)("Bottom"-k_t(hat(bold(f))_t, bold(J)_t) = E_t)$. Rank correlation, measured by Kendall's $tau$@kendall1938measure, quantifies agreement between predicted and actual final rankings. Posterior consistency reflects how concentrated the posterior is: $"Consistency"_t = 1 / N sum_(m=1)^(N) bb(I)(argmin_i "Combined"_i^((m)) in E_t)$. Finally, the effective sample size@gelman2013bayesian assesses chain mixing.
// $
//   "Acc" = 1 / T sum_(t=1)^T bb(I)(text("Bottom")-k_t(hat(bold(f))_t, bold(J)_t) = E_t)
// $

*Elimination Accuracy.*
// 淘汰准确率
The proportion of elimination weeks in which the set of contestants with the $k_t$ lowest combined scores (where $k_t = |E_t|$) equals the actual eliminated set $E_t$. A week is correct only when the predicted bottom-$k_t$ exactly matches the observed eliminations.
// 淘汰周中，模型预测的“综合得分最低的 k_t 名”与实际被淘汰集合 E_t 完全一致的周数占比；k_t = |E_t| 随周次可不同（单淘汰、双淘汰等）。
$
"Acc" = 1 / T sum_(t=1)^T bb(I)(text("Bottom")-k_t(hat(bold(f))_t, bold(J)_t) = E_t)
$


/ Rank Correlation: The agreement between the predicted ranking and the actual ranking is measured by Kendall's $tau$@kendall1938measure (rank correlation),
$
  tau = frac(2(C - D), n(n - 1)),
$
where $C$ is the number of concordant pairs (same order in both rankings), $D$ is the number of discordant pairs, and $n$ is the number of ranked contestants. Values of $tau$ closer to 1 indicate stronger agreement between predicted and actual rankings.

// 后验一致性
/ Posterior Consistency: For each elimination week, the proportion of MCMC samples for which the predicted lowest combined score falls within the actual eliminated set $E_t$; it reflects how concentrated the posterior is for that week.
// 每周淘汰中，MCMC 样本里“预测综合得分最低者属于实际被淘汰集合 E_t”的比例。
$
  "Consistency"_t = 1 / N sum_(m=1)^(N) bb(I)(argmin_i "Combined"_i^((m)) in E_t)
$

// 有效样本量
/ Effective Sample Size: The number of effectively independent draws in the chain; low ESS indicates strong autocorrelation and suggests that posterior intervals should be interpreted with caution. @gelman2013bayesian@brooks2011handbook
$
  "ESS" = N / (1 + 2 sum_(h=1)^infinity rho_h)
$
where $rho_h$ denotes the autocorrelation coefficient at lag $h$.
// 其中 ρ_h 为滞后 h 的自相关系数。

== Results Analysis

#figure(
  image("img/elimination_accuracy_by_season.png", width: 82%),
  caption: [Elimination accuracy by season.],
) <fig:elim-acc>

The model achieves an elimination accuracy of *93.49%* (244 out of 261 elimination weeks), correctly predicting which contestants would be eliminated under the inferred fan vote distribution (@fig:elim-acc). The predicted final rankings exhibit strong agreement with observed outcomes, with Kendall's *$tau = 0.994$* across all seasons; the model correctly identifies the champion in all 34 seasons and recovers the entire top-three ordering in *100% of cases*. Posterior consistency averages *0.90* (std. 0.22), and the mean effective sample size of *191* (minimum 116) confirms adequate chain mixing.


= Task 2: Aggregation Comparison and Recommendation

// RQ1 比较两种方法的结果，是否有一种方法似乎比另一种更偏向粉丝投票
== Comparison of Rank and Percentage Methods

Using the posterior fan-vote distribution from Task 1 and the observed judges' scores, we simulate each week under both aggregation rules: the Rank-based method and the Percentage-based method. For every contestant-week, we obtain fan shares from the MCMC posterior and combine them with the actual judge scores to compute rank-sum and percentage-sum; the contestant(s) with the lowest combined score in that week are taken as eliminated under the corresponding rule. This yields, for each week with elimination, two (possibly different) eliminated contestants—one under Rank and one under Percentage. We then use two indicators to compare the two methods: whether they disagree on who is eliminated (DiffProb), and which rule is more biased toward fan voting (Judge-Favorite Elimination Gap, JFEG).

*DiffProb.*
We define the probability that the two methods disagree on the eliminated contestant in week $t$ as
$
  text("DiffProb")_t = 1/S sum_(s=1)^S bb(I)(e_t^(text("rank"),(s)) eq.not e_t^(text("pct"),(s)))
$
where $e_t^(text("rank"),(s))$ and $e_t^(text("pct"),(s))$ denote the contestant eliminated in week $t$ under the Rank and Percentage rules in simulation $s$, and $S$ is the number of (posterior or bootstrap) runs. With a single run per week, $text("DiffProb")_t = 1$ if the two rules eliminate different contestants and $0$ otherwise. A higher mean DiffProb indicates that the choice of aggregation rule more often changes who is eliminated.

*Judge-Favorite Elimination Gap (JFEG).*
To assess which rule is more biased toward fan voting, we use a subgroup-based measure grounded in conditional expectation. Define the subgroup $H_t$ as contestants who are *judge-favored* (top $q_J$ in judge rank) and *fan-disfavored* (bottom $q_F$ in fan rank) in week $t$—i.e., judge favorites that fans do not like. Let
$
  macron(P)_t^(m) = (1/abs(H_t)) sum_(i in H_t) P(text("eliminated") = i | text("method") = m)
$
be the elimination probability in $H_t$ under method $m$ (with a single realization per week, $macron(P)_t^(m)$ is $1/abs(H_t)$ if the contestant eliminated under $m$ lies in $H_t$, and $0$ otherwise). The *Judge-Favorite Elimination Gap* in week $t$ is
$
  Delta_t^text("JFEG") = macron(P)_t^(text("pct")) - macron(P)_t^(text("rank")).
$
If $Delta_t^text("JFEG") > 0$, the Percentage method eliminates judge favorites (that fans dislike) more often than the Rank method, so we interpret the Percentage method as *more fan-biased* in that week; if $Delta_t^text("JFEG") < 0$, the Rank method is more fan-biased. This comparison is restricted to *disagreement weeks* (weeks where the two methods eliminate different contestants), so that the bias measure is informative. Season-level conclusions are obtained by averaging $Delta_t^text("JFEG")$ over disagreement weeks.

#figure(
  image("img/diffprob_heatmap.pdf", width: 100%),
  caption: [
    Rank vs Percentage.
  ],
) <fig:green_heat>

*Results and conclusions.*
Across 295 weeks with elimination, the two methods yield different eliminated contestants in 30 weeks, giving a mean DiffProb of *10.2%*. Thus in about 90% of weeks the Rank and Percentage rules agree on who is eliminated; in the remaining 10% the choice of rule changes the outcome. On the 30 disagreement weeks, we compute JFEG for the 20 weeks in which the subgroup $H_t$ (judge top 30%, fan bottom 30%) is non-empty. The mean $Delta^text("JFEG")$ over these weeks is *0.675* (positive). Hence on weeks where the two rules actually disagree, the Percentage method eliminates judge favorites that fans dislike more often than the Rank method—i.e., the Percentage method is *more fan-biased* in the sense of JFEG. In sum: (1) the aggregation rule has a material effect in roughly one in ten elimination weeks; (2) when it does, the Percentage-based rule tends to favor fan preference over judge preference more than the Rank-based rule.

// RQ2 四张折线图 4个case的排名变化与波动
== Analysis of Controversial Contestants

In controversial cases (such as Jerry Rice, Billy Ray Cyrus, Bristol Palin, and Bobby Bones), we examine two questions: (1) whether the choice of aggregation method leads to the same result for each contestant; (2) how adding the Judges' Save rule (judges choose which of the bottom two pairs to eliminate) affects outcomes. We present the first part here and the second in the following subsection.

=== Do Both Methods Yield the Same Outcome?

Using the inferred fan votes and the same simulation setup as in Task 2, we compute each contestant's weekly rank under both the Rank-based and Percentage-based methods. The grey shaded region in each panel is the *elimination zone*: in any week, a contestant whose rank falls at or below the cutoff (the dark grey line) would be eliminated under a strict "touch = out" rule; we truncate each series at the first week it touches the zone so that the plot reflects the moment of (simulated) elimination. @fig:controversial-four shows the rank trajectories for the four controversial contestants.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    image("img/controversial_jerry_rice_rank_over_weeks.png", width: 100%),
    image("img/controversial_billy_ray_cyrus_rank_over_weeks.png", width: 100%),
    
    image("img/controversial_bristol_palin_rank_over_weeks.png", width: 100%),
    image("img/controversial_bobby_bones_rank_over_weeks.png", width: 100%),
  ),
  caption: [
    Weekly rank under the Rank method and the Percentage method.
  ],
) <fig:controversial-four>

The two methods do *not* yield the same outcome for every contestant.

- *Jerry Rice (Season 2)* and *Billy Ray Cyrus (Season 4)*: Under both the Rank and Percentage methods, their ranks stay above the elimination cutoff for all weeks shown. Week-by-week ranks differ between the two methods, but neither would be eliminated in the simulated period.

- *Bristol Palin (Season 11)*: Under the Rank method, her rank touches the cutoff in Week 6, so she would be eliminated in that week; the blue series stops there. Under the Percentage method, her rank remains above the cutoff through Week 10, so she would not be eliminated. Thus the choice of method changes her outcome: one rule eliminates her, the other does not.

- *Bobby Bones (Season 27)*: Under the Rank method, his rank touches the cutoff (e.g. in Week 8), implying elimination; the blue series stops at the first touch. Under the Percentage method, he stays above the cutoff and reaches a much better rank by the end of the season. Again, the two methods give different outcomes.


=== Effect of the Judges' Save rule

Under the Judges' Save rule (introduced in Season 28), the show first identifies the bottom two contestants by combined score; judges then choose which one to eliminate. So the rule adds a second step: instead of the lowest-ranked contestant leaving automatically, judges pick one of the bottom two to leave and the other to stay.

None of the four controversial contestants competed under this rule, but we can still say what the rule would do in their situations. For example, under the Rank method Bristol Palin would be in the bottom two in Week 6; under the Percentage method she would not. If Judges' Save had been in place that week, the bottom two would have been determined by combined score, and judges would then have chosen who to eliminate. They could in principle have saved her—or the other contestant. Similarly, Bobby Bones would have been at risk under the Rank method in later weeks; Judges' Save would have given judges the option to save him or his bottom-two counterpart. So in principle, adding the rule can change who goes home whenever the bottom two are identified and judges exercise discretion.

In practice, Task 1 shows (@fig:judges-save) that in Seasons 28–34 judges saved the contestant with higher estimated fan share *90.5%* of the time; they did not systematically protect the contestant with better judge scores. So for controversial contestants who are strong in fan support but weaker in technical rank (like Bristol Palin or Bobby Bones in some weeks), the rule could in principle let judges save them when they land in the bottom two—but the observed behavior suggests judges have tended to align with fan preference rather than override it. In short: adding "judges choose which of the bottom two to eliminate" introduces discretion that can change outcomes, but in the data so far that discretion has not been used to favor technical merit over popularity.

#figure(
  image("img/judges_save_preference.png", width: 82%),
  caption: [
    Judges' Save preference in Seasons 28–34.
  ],
) <fig:judges-save>


// RQ3
== Recommendation
Based on the evidence from Task 1 and Task 2, we offer the following recommendations.

*Aggregation method.*
Across 295 elimination weeks, the Rank and Percentage methods disagree on who is eliminated in about 10% of weeks (mean DiffProb *10.2%*); in the remaining 90% the two rules give the same outcome. When they do disagree, the Judge-Favorite Elimination Gap (JFEG) indicates that the Percentage method eliminates judge favorites that fans dislike more often than the Rank method (mean $Delta^text("JFEG")$ *0.675* on disagreement weeks)—i.e., the Percentage method is more fan-biased in those weeks. Thus the Rank-based method is *less* fan-biased in the minority of weeks where the choice of rule matters. We recommend *adopting the Rank-based aggregation method* if the goal is to temper fan influence in close-call weeks while keeping the same overall structure; if institutional continuity with recent seasons (many of which used percentage-based rules) is the priority, either method is acceptable given the high agreement rate.

*Judges' Save rule.*
Task 1 shows that in Seasons 28–34 judges saved the contestant with higher estimated fan share 90.5% of the time and did not significantly favor the contestant with better judge scores ($p = 0.56$). So the rule, as implemented, has not corrected for fan dominance. We recommend *not relying on the current Judges' Save rule* to protect technical merit. If the rule is retained, it should be *reformed*—e.g., with explicit criteria that require judges to justify saving the contestant with stronger technical performance when the bottom two differ in judge scores—so that it can fulfill its intended role.

= Task 3: Dual-Track Mixed-Effects Model for Impact Analysis
// In this section，我们旨在进一步区分评委评分与粉丝投票这两种不同评价机制，并系统分析职业舞伴与明星个体特征在其中所发挥的作用。由于评委评分与粉丝投票在形成机制与行为逻辑上存在本质差异，若将二者合并建模，将掩盖关键结构性信息。因此，我们构建双轨分层模型，分别对评委评分与粉丝投票进行建模，并在统一框架下进行比较分析。
In this section, we distinguish between the two evaluation mechanisms—judges' scores and fan votes—and systematically analyze the roles of professional partners and celebrity characteristics in each. Because these outcomes differ fundamentally in their formation mechanisms, modeling them together would obscure important structural information. We therefore build a dual-track mixed-effects model@Gelman_Hill_2006@raudenbush2002hierarchical that models each outcome separately and compares them within a unified framework.

== Data Structure and Variable Definition
// 数据结构与变量定义 TODO 标题修改
// 研究数据涵盖第 1 至第 34 季《与星共舞》的周度比赛记录。每一条观测对应某位明星在某一比赛周的表现结果。由于同一位明星会在多个比赛周中重复出现，同时多位明星可能搭档同一位职业舞伴，数据呈现出明显的层级结构。
The data consist of weekly competition records from Seasons 1–34. Each observation is one celebrity's outcome in one competition week. The same celebrity appears in multiple weeks, and multiple celebrities may share the same professional partner, so the data have a clear hierarchical structure.

// 在模型中，我们将周度观测视为嵌套于明星个体之中，而将职业舞伴视为跨明星的高层分组因素。设 $i$ 表示明星，$t$ 表示比赛周次，$p(i)$ 表示明星 $i$ 所对应的职业舞伴。

We treat weekly observations as nested within celebrities and professional partners as a cross-celebrity grouping factor. The definitions can be found in @tab:notations.

// We treat weekly observations as nested within celebrities and professional partners as a cross-celebrity grouping factor. Let $i$ index the celebrity, $t$ the competition week, and $p(i)$ the professional partner of celebrity $i$. We define:

// - $Y_(i,t)^J$: judges' score received by celebrity $i$ in week $t$;
// - $Y_(i,t)^F$: fan-vote outcome for celebrity $i$ in week $t$;
// - $bold(X)_i$: celebrity-level covariate vector (e.g., age, industry);
// - $"Week"_t$: week index, included to capture systematic change over the season.

// 主要变量定义如下：
// - $Y^J_{i,t}$：明星 $i$ 在第 $t$ 周获得的评委评分；
// - $Y^F_{i,t}$：明星 $i$ 在第 $t$ 周获得的粉丝投票结果（标准化后）；
// - $\mathbf{X}_i$：明星层面的个体特征变量向量（如年龄、职业背景等）；
// - $Week_t$：比赛周次，用于控制赛季推进带来的系统性变化。



== Dual-Track Hierarchical Model
// 鉴于评委评分主要反映舞蹈技术水平，而粉丝投票更多体现观众偏好与人气因素，我们分别构建两条模型轨道，在结构上保持可比性，但允许参数与随机效应存在差异。

Judges' scores mainly reflect technical quality; fan votes reflect audience preference and popularity. We therefore specify two linear mixed-effects models with the same structure but separate fixed and random effects. The structure of our idea can be illustrated by @fig:task3-schematic. For judges' scores (Track 1):
// 评委评分模型被设定为线性混合效应模型，其形式为：
$
  Y_(i,t)^J
  =
  beta_0^J
  + bold(beta)^J dot.op bold(X)_i
  + beta_w^J "Week"_t
  + u_(p(i))^J
  + epsilon_(i,t)^J,
$
// where $beta_0^J$ is the fixed intercept, $bold(beta)^J$ gives the effect of celebrity characteristics on judges' scores, and $beta_w^J$ captures the trend over weeks. The partner random effect $u_(p(i))^J tilde cal(N)(0, sigma_u^2)$ captures systematic differences across partners; $epsilon_(i,t)^J$ is the residual with variance $sigma_epsilon^2$. For fan votes (Track 2), we keep the same structure but add the current-week judges' score:
For fan votes (Track 2), we keep the same structure but add the current-week judges' score:
$
  Y_(i,t)^F
  =
  beta_0^F
  + bold(beta)^F dot.op bold(X)_i
  + gamma Y_(i,t)^J
  + u_(p(i))^F
  + epsilon_(i,t)^F.
$

// 其中，$\gamma$ 衡量评委评分对粉丝投票的边际影响，随机效应 $u_{p(i)}^F \sim \mathcal{N}(0,\sigma_F^2)$ 则用于刻画职业舞伴在粉丝投票层面所产生的长期影响。
// The coefficient $gamma$ measures the marginal effect of judges' score on fan vote. The partner random effect $u_(p(i))^F tilde cal(N)(0, sigma_u^2)$ captures the long-run effect of the partner on fan outcomes; $epsilon_(i,t)^F$ is the residual. Each track has its own $sigma_u^2$ and $sigma_epsilon^2$.
The coefficient $gamma$ measures the marginal effect of judges' score on fan vote.

#figure(
  image("img/task3_flowchart.png", width: 60%),
  caption: [
    Schematic of Dual-Track Hierarchical Model
  ],
) <fig:task3-schematic>

== Evaluation Metrics

To answer how professional partners and celebrity characteristics affect judges' scores and fan votes, we employ three evaluation metrics. 

// The Intraclass Correlation Coefficient (ICC)@shrout1979intraclass measures the fraction of outcome variance explained by the professional partner: $"ICC" = sigma_u^2 / (sigma_u^2 + sigma_epsilon^2)$. The coefficient $gamma$ in the fan-vote model represents the performance-to-popularity conversion rate. Finally, the Impact Divergence Index (IDI) quantifies whether a covariate affects the two tracks differently: $"IDI"_k = (|beta_k^J - beta_k^F|) / sqrt("Var"(beta_k^J) + "Var"(beta_k^F))$, where values exceeding 1.96 indicate significant divergence.

// 职业舞伴在某一轨道中解释的结果变异比例。ICC 越大，说明职业舞伴在该轨道中的系统性影响越强；比较两条轨道的 ICC 可直接判断职业舞伴是更偏“技术教练”还是更偏“流量操盘手”。
*Intraclass Correlation Coefficient (ICC)*: The ICC is the fraction of outcome variance in each track explained by the professional partner@shrout1979intraclass:
$
  "ICC" = frac(sigma_u^2, sigma_u^2 + sigma_epsilon^2),
$
// where $sigma_u^2$ is the partner random-effect variance and $sigma_epsilon^2$ is the residual variance in that track. A larger ICC in one track indicates a stronger systematic effect of the partner on that outcome; comparing ICC between the judges' and fan-vote models therefore reveals whether partners matter more for technical scores or for audience votes.
A larger ICC in one track indicates a stronger systematic effect of the partner on that outcome; comparing ICC between the judges' and fan-vote models therefore reveals whether partners matter more for technical scores or for audience votes.

// 在控制明星特征与周次后，评委分数对粉丝投票的边际影响。若 γ > 0 且显著，说明粉丝投票随本周表现提升（“表现驱动”）；若 γ ≈ 0 且不显著，说明粉丝投票主要由长期人气与组合效应驱动（“身份驱动”）。
*Performance Conversion Rate ($gamma$)*: The coefficient $gamma$ in the fan-vote model is the marginal effect of the current-week judges' score on fan vote, after controlling for celebrity characteristics and week.

A significant positive $gamma$ implies that fan votes respond to weekly performance (“performance-driven”); if $gamma approx 0$ and is not significant, fan votes are driven mainly by long-run popularity and pairing effects, and single-week technical changes have little impact (“identity-driven”).

// 同一特征在评委轨道与粉丝轨道中的系数差异（经标准误标准化）。IDI 大且显著说明该特征对评委分数与粉丝投票的影响不一致（方向或大小不同）；IDI 小且不显著说明两条轨道对该特征的反应一致。
*Impact Divergence Index (IDI)*: For each covariate (e.g., industry, age), the Impact Divergence Index is the standardized difference between its coefficient in the judges' model and its coefficient in the fan-vote model:
$
  "IDI"_k = (|beta_k^J - beta_k^F|) / sqrt("Var"(beta_k^J) + "Var"(beta_k^F))
$
A large and significant IDI for a covariate indicates that the covariate affects judges' scores and fan votes differently (in direction or magnitude); a small or non-significant IDI indicates that the two tracks respond to that characteristic in a similar way.

The Best Linear Unbiased Prediction (BLUP)@henderson1975best of the partner random effect yields $hat(u)_(p(i))^J$ (technical boost) and $hat(u)_(p(i))^F$ (popularity boost), which we use to classify partners in a technology–popularity plane.

== Results Analysis
// 估计 TODO
// 两条模型轨道均采用线性混合效应回归进行估计。职业舞伴被视为随机截距项，以刻画其跨明星的系统性影响。明星职业背景等分类变量以演员类别作为基准组进行编码。

// 通过在同一数据集上分别估计两条模型，并对关键参数进行对比分析，我们能够识别明星特征与职业舞伴在评委评分与粉丝投票中的差异化作用机制。
We estimate both tracks by linear mixed-effects regression. The professional partner enters as a random intercept. Categorical covariates (e.g., industry) are dummy-coded with the actor category as the reference. By fitting the two models on the same dataset and comparing coefficients and ICCs, we identify how celebrity characteristics and partners affect judges' scores versus fan votes.

#figure(
  image("img/variance_decomposition.png", width: 75%),
  caption: [
    Variance explained by professional partner (ICC by track)
  ],
) <fig:task3-var>

// *组内相关系数（ICC）。*拟合模型显示，评委组的ICC值约为9.5%，粉丝投票组的ICC值约为21.8%（见图@fig:task3-var）。因此，搭档对观众投票的影响程度大于对技术评分的解释力，这意味着选择专业搭档对人气的影响远大于对评委评分的决定作用。

*Intraclass correlation (ICC)*: The fitted models yield ICC $approx 9.5%$ for the judges' track and ICC $approx 21.8%$ for the fan-vote track (@fig:task3-var)—a 2.3-fold difference indicating that partner choice is more consequential for popularity than for technical performance.

#figure(
  image("img/coefficient_comparison.png", width: 90%),
  caption: [
    Fixed-effect coefficients in the judges' model vs. the fan-vote model
  ],
) <fig:task3-coef>

// *影响差异（IDI）。* 多个协变量呈现显著且显著的IDI。*周*因素呈现最大差异（IDI约27.3，p<0.001）：评委评分随赛季推进显著上升（β约0.31），而粉丝投票占比反应较弱（β约0.11），这与选手淘汰后投票集中化的现象相符。*年龄*在两组数据中均为负相关，但评委组更显著（IDI约17.2，p<0.001）。最显著的是，*运动员身份*（相对于演员身份）呈现相反效应：评委模型中$β^J≈-0.129$（$p≈0.017$），而粉丝投票模型中$β^F≈+0.052$（$p<0.001$）（见图@fig:task3-coef）。这种“运动员悖论”表明，同一特征既可能降低技术分又可能提升粉丝支持度，说明评委与粉丝对名人特质的权重评估存在差异。

*Impact divergence (IDI)*: Several covariates show large and significant IDI. Week exhibits the largest divergence (IDI $approx 27.3$, $p < 0.001$): judges' scores rise strongly over the season ($beta approx 0.31$) while fan-vote share responds less ($beta approx 0.11$), consistent with vote concentration as contestants are eliminated. Age is negative in both tracks but much stronger for judges (IDI $approx 17.2$, $p < 0.001$). Most notably, Athlete (vs. Actor) exhibits opposite signs: $beta^J approx -0.129$ ($p approx 0.017$) in the judges' model and $beta^F approx +0.052$ ($p < 0.001$) in the fan-vote model (@fig:task3-coef). This "athlete paradox" demonstrates that the same characteristic can reduce technical scores while increasing fan support—judges and fans do not weight celebrity characteristics in the same way.

// *表现转化率（$gamma$）。* 在粉丝投票模型中，评委评分的估计系数为 $gamma ≈ 0.0039$（标准误 $≈ 0.0055$，$p ≈ 0.47$）。在控制名人效应和搭档效应后，每周评委评分每变化一个单位，粉丝投票份额不会出现显著变动。因此该竞赛属于“身份驱动型”而非“表现驱动型”：选手基本盘形成后粉丝支持度基本稳定，短期技术性波动对投票结果的边际影响微乎其微。

*Performance conversion rate ($gamma$)*: The raw coefficient of judges' score in the fan-vote model is $gamma approx 0.004$ (SE $approx 0.005$, $p approx 0.42$), appearing non-significant. However, sensitivity analysis reveals substantial multi-collinearity: when week is removed from the model, $gamma$ rises to 0.161 ($p < 0.001$). This occurs because week and judges' score are positively correlated due to survivor bias—contestants who persist longer tend to have higher scores. The interaction term $gamma times "week"$ is also significant ($p < 0.001$), indicating that the performance-to-popularity conversion strengthens as the season progresses. We conclude that fans do respond to technical performance, but this effect is confounded with longevity in the original specification. The correct interpretation is that "fans conflate longevity with quality"—they reward both, but cannot easily distinguish between the two.

#figure(
  image("img/pro_scatter.png", width: 85%),
  caption: [
    Professional partners in the technology–popularity plane
  ],
) <fig:task3-pro>

// *搭档角色*。通过结合两条赛道的最佳线性预测值（BLUPs），可将搭档分为四类（图 @fig:task3-pro）。技术与人气效应俱佳者（如德里克·霍夫、乔纳森·罗伯茨、马克西姆·切梅尔科斯基、托尼·多沃拉尼、凯姆·约翰逊）扮演“造王者”角色；其余选手（如阿尔捷姆·奇格温采夫、瓦伦丁·切梅尔科夫斯基）则展现出强劲技术实力但人气效应较弱（“技术型选手”）。这印证了核心结论：伴侣选择对动员粉丝支持的重要性，远高于提升评委评定的表现水平。

*Partner roles*: Using the BLUPs from both tracks, partners fall into four quadrants (@fig:task3-pro). Those with high technology and high popularity effects—Derek Hough, Jonathan Roberts, Maksim Chmerkoskiy, Tony Dovolani, Kym Johnson—act as "kingmakers," while others such as Artem Chigvintsev and Valentin Chmerkovskiy show strong technical but weaker popularity effects ("technicians"). This supports the conclusion that partner choice matters more for mobilizing fan support than for improving judge-assessed performance.

= Task 4: Progressive Technical Fairness System for DWTS

// 最后的挑战要求我们"提出另一种每周结合粉丝投票与评委分数的系统，使其更'公平'（或在其他方面更好，如让比赛对粉丝更刺激）"。本节综合运用 Task 1 的粉丝份额估计和 Task 3 的影响因素分析，提出一个新的投票组合系统——渐进技术公平系统（PTFS）。

The final challenge requires us to _"propose another system using fan votes and judge scores each week that you believe is more 'fair' (or 'better' in some other way such as making the show more exciting for the fans)."_ This section synthesizes the fan share estimates from Task 1 and the impact factor analysis from Task 3 to propose a new vote aggregation system—the *Progressive Technical Fairness System (PTFS)*.

== Motivation and Problem Diagnosis

Our analyses from Tasks 1 and 3 provide a rigorous diagnostic foundation for system redesign. Task 1 revealed that the Judges' Save mechanism introduced in Season 28 has largely failed to protect technically superior contestants: in 90.5% of cases, judges saved the contestant with higher estimated fan share rather than the one with better technical skills ($p = 0.56$, binomial test). Meanwhile, Task 3 uncovered what we term the "Athlete Paradox"—athletes receive lower judge scores ($beta^J = -0.129$, $p = 0.017$) yet higher fan votes ($beta^F = +0.052$, $p < 0.001$), revealing systematic divergence between the two evaluation mechanisms (IDI = 3.26). Furthermore, professional dancers explain 2.3 times more variance in fan voting (ICC = 21.8%) than in judge scoring (ICC = 9.5%), indicating that partner assignment disproportionately influences popularity outcomes.

These findings suggest that neither a pure judge-based system (which carries its own biases) nor a pure fan-based system (which prioritizes celebrity recognition over dance skill) can achieve fairness in isolation. The challenge is to design a hybrid system that progressively balances these competing objectives.

== Design Principles

Drawing from social choice theory@arrow1951social, we identify three desirable properties for any improved voting system. First, *technical fairness* requires that final rankings correlate positively with cumulative technical performance, ensuring skilled dancers are appropriately rewarded. Second, *audience engagement* demands that the system maintain competitive tension and meaningful fan participation, preserving the show's entertainment value. Third, *robustness* requires consistent performance across different seasons and contestant compositions, avoiding pathological outcomes.

These properties inherently admit trade-offs. Maximizing technical fairness through 100% judge scores would eliminate fan engagement entirely, while pure fan voting—as demonstrated by the Bobby Bones controversy in Season 27—can produce outcomes widely perceived as unfair. We therefore propose a Progressive Technical Fairness System (PTFS) that dynamically adjusts the balance between judge scores and fan votes across the competition timeline.

== Model Specification

Let $n_t$ denote the number of contestants competing in week $t$, and let $T$ be the total number of weeks in the season. For contestant $i$ in week $t$, we define $J_(i,t)$ as the average judge score (normalized to $[0, 10]$) and $hat(f)_(i,t)$ as the estimated fan share from Task 1's Bayesian MCMC model. The PTFS combined score is computed as:
$
  S_(i,t) = w_J (t) dot.op tilde(J)_(i,t) + (1 - w_J (t)) dot.op tilde(F)_(i,t)
$

where $tilde(J)_(i,t)$ and $tilde(F)_(i,t)$ are normalized scores within each week:
$
  tilde(J)_(i,t) = frac(J_(i,t), sum_(j=1)^(n_t) J_(j,t)), quad
  tilde(F)_(i,t) = frac(hat(f)_(i,t), sum_(j=1)^(n_t) hat(f)_(j,t))
$

This normalization ensures both components sum to unity within each week, enabling meaningful combination regardless of absolute magnitudes.

The key innovation of PTFS is a time-varying judge weight defined by
$
  w_J (t) = w_J^"start" + (w_J^"end" - w_J^"start") dot.op frac(t - 1, T - 1)
$

This linear interpolation produces $w_J = w_J^"start"$ in Week 1 (lower judge weight, higher fan influence) and $w_J = w_J^"end"$ in the final week (higher judge weight, technical merit emphasized). Through grid search over 34 seasons, we identified optimal parameters of $w_J^"start" = 0.45$ and $w_J^"end" = 0.80$.

#figure(
  image("img/dynamic_weights_professional.png", width: 85%),
  caption: [
    Dynamic weight progression in PTFS: Judge weight increases linearly from 45% (Week 1) to 80% (Final Week), while fan weight decreases correspondingly from 55% to 20%.
  ],
) <fig:dynamic-weights>

The rationale behind this progression reflects the different priorities at each stage of the competition. In the early weeks, with judge weight at 45%, fan votes carry more influence, maintaining suspense and engagement while allowing "underdogs" to survive. As the competition progresses toward the finale, judge weight rises to 80%, ensuring that technical scores dominate and that finalists are genuinely skilled dancers who have earned their place through demonstrated ability.

== Evaluation Framework

To rigorously evaluate PTFS against existing systems (Rank-based and Percentage-based), we developed a multi-dimensional evaluation framework comprising technical fairness metrics, engagement metrics, and robustness measures.

#figure(
  table(
    columns: (0.9fr, 2fr, auto),
    align: (left, left, center),
    table.hline(),
    [*Metric*], [*Definition*], [*Direction*],
    table.hline(),
    [Kendall's $tau$], [Correlation between cumulative judge ranks and final placements], [Higher (↑)],
    [Spearman's $rho$], [Rank correlation (alternative measure)], [Higher (↑)],
    [Tech Top3 → Final Top3], [Proportion of technical top-3 who finish in final top-3], [Higher (↑)],
    [Tech Lowest Eliminated], [Proportion of weeks where technical lowest is eliminated], [Higher (↑)],
    [Mean Rank Deviation], [Average absolute difference between technical and final rank], [Lower (↓)],
    table.hline(),
  ),
  caption: [Technical fairness metrics for voting system evaluation.],
) <tab:fairness-metrics>

Kendall's $tau$ serves as our primary fairness metric, measuring the ordinal association between cumulative technical performance and competitive outcomes:

$
  tau = frac("concordant pairs" - "discordant pairs", binom(n, 2))
$

#figure(
  table(
    columns: (1fr, 2fr, auto),
    align: (left, left, center),
    table.hline(),
    [*Metric*], [*Definition*], [*Interpretation*],
    table.hline(),
    [Close Call Rate], [Proportion of eliminations with score gap < 2%], [Higher = more suspense],
    [Tech Top50 Eliminated], [Proportion of "upset" eliminations], [Moderate = balanced],
    [$tau$ Std Across Seasons], [Standard deviation of $tau$ across 34 seasons], [Lower = more robust],
    [Consistent Seasons], [Proportion of seasons with $tau > 0.5$], [Higher = reliable],
    table.hline(),
  ),
  caption: [Engagement and robustness metrics.],
) <tab:engagement-metrics>

== Experimental Results

We simulated all three voting systems across 34 seasons of DWTS (1,411 contestant-weeks, 261 elimination events) using estimated fan shares from Task 1. @tab:system-comparison presents the main comparison results.

#figure(
  table(
    columns: (2fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    table.hline(),
    [*Metric*], [*Rank-based*], [*Percentage*], [*PTFS*], [*Best*],
    table.hline(),
    table.cell(colspan: 5, align: left)[_Technical Fairness_],
    [Kendall's $tau$], [0.674], [0.727], [*0.749*], [PTFS],
    [Spearman's $rho$], [0.815], [0.860], [*0.876*], [PTFS],
    [Tech Top3 → Final Top3], [59.8%], [71.6%], [*72.5%*], [PTFS],
    [Mean Rank Deviation], [1.61], [1.35], [*1.26*], [PTFS],
    [Tech Lowest Eliminated], [41.9%], [47.2%], [*52.8%*], [PTFS],
    table.hline(),
    table.cell(colspan: 5, align: left)[_Engagement_],
    [Close Call Rate], [30.2%], [66.1%], [*79.1%*], [PTFS],
    [Tech Top50 Eliminated], [7.0%], [9.6%], [8.6%], [Balanced],
    table.hline(),
    table.cell(colspan: 5, align: left)[_Robustness_],
    [$tau$ Std Across Seasons], [0.150], [0.129], [*0.125*], [PTFS],
    [Consistent Seasons], [91.2%], [94.1%], [*97.1%*], [PTFS],
    table.hline(),
  ),
  caption: [System comparison across all metrics (34 seasons). PTFS outperforms both existing methods across all primary metrics.],
) <tab:system-comparison>

As shown in @tab:system-comparison, PTFS outperforms both existing systems across all primary metrics. The improvement is most pronounced in technical fairness, where PTFS achieves a +0.022 increase in $tau$ over the Percentage-based method (a 3.0% relative improvement). Engagement also improves substantially, with the Close Call Rate rising by 13.0 percentage points to 79.1%. In terms of robustness, 97.1% of seasons under PTFS achieve $tau > 0.5$, compared to 94.1% for the Percentage-based system. The improvement of PTFS over the Percentage-based system is statistically significant ($p = 0.031$, paired $t$-test across 34 seasons), providing strong evidence that PTFS represents a genuine improvement rather than random variation (@tab:significance).

#figure(
  table(
    columns: (2fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    table.hline(),
    [*Comparison*], [$Delta tau$], [*Test Statistic*], [$p$*-value*], [*Significant*],
    table.hline(),
    [PTFS vs. Percentage], [+0.022], [$t_(33) = 2.26$], [*0.031*], [✓ Yes],
    [PTFS vs. Rank-based], [+0.075], [$t_(33) = 4.12$], [*< 0.001*], [✓ Yes],
    table.hline(),
  ),
  caption: [Paired statistical tests (PTFS vs. alternatives).],
) <tab:significance>

#figure(
  image("img/tau_by_season.png", width: 95%),
  caption: [
    Kendall's $tau$ comparison across 34 seasons.],
) <fig:tau-by-season>

== Trade-off Analysis

No voting system can simultaneously maximize all desirable properties—this is a fundamental insight from social choice theory@arrow2020social. PTFS represents a principled trade-off:

#figure(
  table(
    columns: (1.4fr, 1fr, 3fr),
    align: (left, center, left),
    table.hline(),
    [*Dimension*], [*Change*], [*Interpretation*],
    table.hline(),
    [Technical Fairness ($tau$)], [*+0.022*], [Statistically significant improvement ($p = 0.031$)],
    [Ranking Consistency], [*+2.9%*], [97% vs. 94% seasons with $tau > 0.5$],
    [Suspense (Close Call)], [*+13.0%*], [Substantially more "edge-of-seat" eliminations],
    [Fair Elimination], [*+5.6%*], [Tech lowest eliminated more often (52.8% vs. 47.2%)],
    [Upset Rate], [*-1.0%*], [Slightly fewer "unfair" eliminations],
    table.hline(),
  ),
  caption: [PTFS trade-off analysis vs. Percentage-based system.],
) <tab:tradeoff>

The trade-off analysis demonstrates that PTFS achieves a superior systemic balance compared to the Percentage-based system. It delivers statistically significant improvements in technical fairness, ranking consistency, suspense, and elimination fairness, while maintaining a comparable level of audience engagement (as reflected in the similar upset rate). This represents a more desirable overall performance profile for competitive reality TV formats.

#figure(
  image("img/improvement_distribution.png", width: 85%),
  caption: [
    Distribution of $tau$ improvement (PTFS $-$ Percentage) across 34 seasons.
  ],
) <fig:improvement-dist>

== Sensitivity Analysis

To assess the robustness of our parameter choices, we conducted a comprehensive sensitivity analysis over the parameter space.

#figure(
  table(
    columns: (1.5fr, 2fr, 1fr),
    align: (center, center, center),
    table.hline(),
    [*Parameter*], [*Search Range*], [*Optimal*],
    table.hline(),
    [$w_J^"start"$], [[0.35, 0.40, 0.45, 0.50]], [*0.45*],
    [$w_J^"end"$], [[0.60, 0.65, 0.70, 0.75, 0.80]], [*0.80*],
    [Improvement bonus $alpha$], [[0.00, 0.05, 0.10, 0.15]], [*0.00*],
    [Protection threshold], [[0.00, 0.10, 0.15, 0.20]], [*0.00*],
    table.hline(),
  ),
  caption: [Parameter grid search results. The optimal configuration uses dynamic weights without additional bonus or protection mechanisms.],
) <tab:sensitivity>

The grid search evaluated 320 parameter combinations, optimizing a composite objective function weighted toward technical fairness (40%), consistency (15%), tech winner rate (15%), top-3 retention (10%), and moderate upset rate (5%). A key finding from this analysis is that additional mechanisms such as improvement bonuses and protection thresholds show diminishing or negative returns when increased from zero, suggesting that the core dynamic weight mechanism alone is sufficient and that additional complexity does not improve performance.

= Strengths and Weaknesses

== Strengths
// 不可观测投票行为的间接推断能力
// 本文构建的贝叶斯推断框架将 DWTS 官方淘汰规则嵌入似然函数，通过不等式约束对粉丝投票这一不可观测变量进行反演。该方法无需引入真实投票数据，即可在制度约束下实现参数识别，从而有效提升了模型在数据受限条件下的可行性与推断稳定性。
(1) We develop a Bayesian inference framework that embeds the official DWTS elimination rules into the likelihood function, using inequality constraints to infer latent fan vote shares. This approach enables parameter identification without access to raw voting data, thereby improving feasibility and inferential stability under limited information.

// 通过构建评委评分与粉丝投票的双轨混合效应模型，本文避免了将技术评价与人气偏好混合建模所带来的结构性偏误。该建模策略能够在统一框架下定量比较职业舞伴与明星特征在两种评价机制中的差异化影响，从而更准确地揭示比赛结果的形成机制。
(2) By modeling judges' scores and fan votes in separate hierarchical tracks, we avoid structural bias arising from conflating technical evaluation with popularity. This framework allows a quantitative comparison of how professional partners and celebrity attributes affect the two evaluation mechanisms differently, yielding a clearer understanding of outcome formation.

// **公平性目标的可操作化与可验证性**
//   提出的 PTFS 通过动态调整评委与粉丝权重，在技术公平性、观众参与度与鲁棒性之间实现可量化的权衡，并通过多维指标与统计检验验证其改进效果。
(3) The proposed PTFS translates fairness considerations into an explicit, testable mechanism by dynamically adjusting the weights of judges' scores and fan votes. Its performance is assessed using multiple quantitative metrics and statistical tests, ensuring that fairness improvements are empirically verifiable rather than qualitatively claimed.

== Weaknesses
//   模型假设粉丝投票在相邻周之间平滑变化，虽有助于识别与稳定推断，但可能无法完全刻画突发事件（如争议表现或舆论风向变化）带来的短期剧烈波动。
(1) The model assumes that fan vote shares evolve smoothly across adjacent weeks. While this assumption stabilizes identification and inference, it may not fully capture short-term abrupt fluctuations driven by exogenous shocks, such as controversial performances or sudden shifts in public opinion. Future work could explore state-space models with regime-switching capabilities to capture these jumps explicitly.

// 尽管我们的模型成功地反演出粉丝投票，但其先验信息主要依赖于明星的静态特征（如年龄、行业）。实际上，粉丝投票受到大量外部动态因素的影响，例如选手在社交媒体上的活跃度、当周新闻热度、甚至宏观经济情绪等。由于缺乏这些高质量的外部数据源，我们的模型未能将其纳入考量.
(2) Although the model successfully infers fan votes, its priors rely primarily on static celebrity attributes (e.g., age and industry). In practice, fan voting is influenced by dynamic external factors such as social media activity. The absence of high-quality external data sources limits the precision and explanatory power of the inferred fan vote distributions.