# YOLO_PEFT Paper — 27 pages extracted


--- PAGE 1 ---

Abstract
YOLO-PEFT: Parameter-Efficient Fine-Tuning on YOLO
Family
Xu Lin
1,*
, WenJie Nie
2
, Jinlong Peng
1
, Weifu Fu
1
, YueXiao Ma
2
, Xiawu Zheng
2
Yong Liu
1
,
1
Tencent Youtu Lab
2
Xiamen University
ABSTRACT
Deploying YOLO-family detectors at scale demands task-, domain-, and hardware-specific variants,yet existing adaptation is either prohibitively expensive or structurally fragile: full fine-tuningreplicates and stores an entire model per variant, while PEFT techniques naively transplantedfrom language models can be inserted into detector layers that are structurally unsafe or semanti-cally incompatible, causing silent accuracy collapse or convergence failure. Unlike homogeneousTransformer backbones, modern real-time detectors are heterogeneous execution graphs thatinterleave dense, grouped, and depthwise convolutions, multi-scale necks, decoupled detectionheads, Distribution-Focal-Loss projections, attention and text-fusion modules, RT-DETR decoders,and MoE routers. We propose YOLO-PEFT, a structure-aware PEFT framework that formulatesdetector adaptation as a constrained adapter-placement problem. Given a detector graph, a PEFTrequest, and a resource budget, YOLO-PEFT parses modules into operator and semantic roles,filters out structurally or semantically unsafe targets through a formal feasibility check, assignsbudget-aware ranks, and lowers the resulting plan into a YOLO-compatible train–save–merge–export runtime—rejecting infeasible placements before training rather than blindly attemptingthem. Evaluated on PASCAL VOC object detection across YOLO11s, YOLO12s, and RT-DETR-L variants, YOLO-PEFT achieves4
.
7
×
–23
.
3
×
adapter parameter compression relative to full
fine-tuning, reduces training memory by70%–75%, and saves87
.
7%per-adapter distribution cost
while retaining86
.
1%–113
.
2%of full fine-tuning mAP50-95. We further demonstrate that seven
of seven LoRA/DoRA/LoHa/LoKr variants collapse catastrophically on RT-DETR-L withoutplacement refusal—confirming that safe adapter placement isarchitecture-conditionedrather thanmethod-inherited. YOLO-PEFT turns YOLO adaptation from manual target-module trial anderror into valid, compact, and deployable adapter planning, with an open-source runtime thatsupports ONNX and TensorRT export out of the box.Parameter-Efficient Fine-Tuning, Object Detection, YOLO, Adapter Placement, HeterogeneousArchitectures, Model Adaptation.:Keywords:Project Page:github.com/Tencent/YOLO-Master
1 IntroductionModern YOLO deployments rarely use a single detector unchanged across all scenarios. Each downstreamapplication may require a different class vocabulary, data domain, camera viewpoint, input resolution,latency target, or hardware backend, so fine-tuning is not an occasional post-processing step but a routinerequirement for deploying real-time detectors at scale, as seen across recent Ultralytics YOLO releases,RT-DETR, YOLO-World, YOLO12, and YOLO-Master MoE [
3
,
16
,
17
,
22
,
27
,
41
]. Full fine-tuning is a
simple adaptation path, yet it replicates and stores an entire model per downstream variant, incurring linearlyscaling training costs, multiplied deployment storage, and—in low-data or domain-shifted settings—the riskof overfitting or overwriting useful pretrained representations. Parameter-efficient fine-tuning (PEFT) seems
1

--- PAGE 2 ---

the natural remedy, but existing PEFT interfaces solve the wrong problem for detectors [6, 8, 10, 28, 46].
Existing PEFT implementations, such as HuggingFace PEFT, expose target selection through target-module
names or module-type matching [28], and automatic PEFT search remains largely Transformer-centric [52]—
implicitly assuming a regular block stack dominated by linear projections. YOLO-family detectors violate
this assumption at every level, and the violation is not abstract: a target-module rule that is benign for a
language model—e.g., “target every convolutional and linear layer”—is exactly the kind of default that, on
a detector, silently selects DFL projection layers or regression-head sublayers tied to detection-specific loss
geometry, or expands the adapter past its intended budget through indiscriminate dense placement. More
specifically: at theoperator level, dense, grouped, and depthwise convolutions obey different rank, group, and
kernel constraints [16, 17]; at thesemantic level, DFL projections and decoupled classification/regression
heads are tied to detection-specific losses and output distributions [16, 17, 41]; at thearchitecture level,
attention blocks, open-vocabulary fusion modules, RT-DETR decoders, and MoE experts introduce additional
stability and routing constraints [3, 5, 22, 27, 39, 41]; and at thedeployment level, adapters must remain
compatible with checkpointing, adapter-only persistence, merging, model fusion, ONNX export, and TensorRT
deployment [31, 32]. Consequently, PEFT on YOLO remains an engineering trial-and-error process instead
of a reproducible planning problem.We therefore argue that PEFT for YOLO is not merely a
question of which low-rank parameterization to use.The central question iswherean adapter can
be safely placed, under operator validity, detection-head semantics, graph-interface compatibility, parameter
budgets, and deployment constraints—the central thesis of this paper. Based on this formulation, we propose
YOLO-PEFT, a structure-aware PEFT framework that resolves YOLO-family PEFT as constrained adapter
placement on a heterogeneous detector graph, as summarized in Fig.??. YOLO-PEFT first parses the
detector into a role-aware graph, tagging every node with anoperator role(dense / grouped / depthwise
conv, linear, attention) and asemantic role(backbone, neck, classification head, DFL projection, MoE router,
etc.). A structure-aware planner then filters this candidate set under operator validity, semantic safety,
and graph-interface compatibility, and solves a budgeted rank-assignment problem to produce a placement
plan—or returnsRefuseand falls back to full fine-tuning when no feasible or reliable plan exists. The
resolved plan is lowered into a YOLO-compatible execution contract that bridges generic PEFT wrappers
(LoRA, DoRA, LoHa, LoKr, IA3, HRA, optionally with RS-LoRA scaling) with the Ultralytics training
and deployment pipeline, including feature routing, checkpointing, fusion safety, adapter-only persistence,
merging, and export to ONNX/TensorRT with zero additional inference overhead. Experiments across YOLO-
family architectures show that stable PEFT placement is architecture-conditioned rather than inherited from
LLM-PEFT rankings; under the proposed execution contract, our YOLO-compatible runtime reproduces the
public v26.02 deployment economics [22], achieving4 .7–8.1× on-disk compression,70–75%training-memory
reduction, and87 .7%lower per-adapter distributed training cost. Our contributions are summarized as
follows:
• A new formulation of PEFT of the YOLO-family(addresses the above interface mismatch). We
formulate parameter-efficient fine-tuning for YOLO-style detectors as structure-constrained adapter
placement: the adapter target set must jointly satisfy operator feasibility, detection-head semantics,
graph-interface compatibility, parameter-budget constraints, and deployment constraints—a problem
left unaddressed by generic target-module matching.
• A structure-aware planning and runtime framework(addresses the thesis:where, notwhich).
We design a constraint-resolved planner that parses heterogeneous YOLO graphs into operator/semantic
roles, filters structurally or semantically unsafe targets (e.g., depthwise or DFL-related projections, MoE
routers, RT-DETR sampling offsets), applies architecture-specific inclusion/exclusion rules (attention,
text-fusion, MoE), and performs budget-aware rank assignment—or refuses infeasible/unreliable plans.
The resolved plan is lowered into a YOLO-compatible runtime that preserves feature routing, gradient
checkpointing, fusion safety, and adapter-only save/load/merge/export.
• An empirical finding that adapter stability is architecture-conditioned(addresses the “trial-and-
error” problem raised above). Through a broad study spanning CNN-only, attention-based, text-fusion,
RT-DETR-style, and MoE-based detectors, we show that PEFT variants have no universal stability
ranking across the YOLO family—rankings inherited fromLLM-oriented PEFT fail to transfer—and that
2

--- PAGE 3 ---

refusal, not blind training, is a necessary capability for safe adapter placement. We further reproduce
public deployment-economics results across the YOLO-family release matrix to quantify storage efficiency,
training memory, accuracy retention, and deployment cost.
Overall, YOLO-PEFT turns PEFT on YOLO from a manual target-selection heuristic into a structure-
constrained planning and deployment problem, enabling adapter placement that is valid, compact, and
deployable across heterogeneous YOLO-family detectors.
2 Related Work
Parameter-efficient fine-tuning (PEFT) reduces adaptation cost by freezing most pretrained weights and
learning a compact set of task-specific parameters. LoRA [10] learns low-rank additive updates∆W =
BA; subsequent methods refine the local update rule or training schedule, including DoRA [24], IA3 [23],
AdaLoRA [51], RS-LoRA [18], LoRA+[7], LoRA-FA [50], DyLoRA [42], PiSSA [29], OLoRA [1], OFT [35],
BOFT [25], and HRA [49]. LoHa and LoKr are prominent in LyCORIS-style low-rank customization [48],
and the Hadamard-product parameterisation underlying LoHa traces to FedPara [30]. Surveys and unified
frameworks cover this growing design space [6, 8, 46]. HuggingFace PEFT exposes many of these methods
behind a uniform target-module interface [28]; related systems study adapter composition and serving at
scale [12, 40]. All of these works addresswhat parameterisationto use once a host layer has been selected.
They do not addresswhereadapters should be placed in a heterogeneous detector graph, nor whether the
selected layers satisfy detection-head semantics, graph-interface constraints, parameter budget limits, or
deployment requirements.
On the vision side, PEFT has been studied through visual prompt tuning [13], AdaptFormer [2], Convpass [14],
residual adapters [9, 21, 33, 36], AdapterFusion [34], and related families [11]. These works demonstrate that
lightweight adaptation transfers effectively to visual recognition. Most of them, however, assume a relatively
regular backbone—typically a Transformer-style block stack—and rely on uniform module substitution or
hand-designed adapter insertion points. Automated PEFT search methods similarly operate within this
target-module configuration space [52]. Such assumptions do not transfer directly to YOLO-family detectors,
whose graphs interleave backbone stages, neck fusion paths, decoupled detection heads, fixed DFL projections,
area-attention blocks, text-fusion layers, and MoE experts.
Several recent studies demonstrate that efficient adaptation matters for object detectors specifically. CULoRA
applies LoRA to few-shot, source-free domain adaptive detection in the YOLO family [47]; SF-YOLO tackles
source-free domain adaptation under a teacher–student scheme [43]. These works validate the practical value
of parameter-efficient adaptation for detectors, but are designed around specific adaptation settings. Neither
treats the detector as a heterogeneous graph whose adapter targets must be resolved jointly under operator
validity, semantic safety, graph-interface compatibility, budget feasibility, and deployment constraints.
Modern real-time detectors are no longer uniform convolutional stacks. The YOLO family has evolved
from early one-stage architectures to diverse multi-scale systems with industrial and end-to-end variants [15–
17, 19, 37, 38, 44, 45]. RT-DETR integrates a Transformer-style detection decoder [27]; YOLO-World
adds open-vocabulary text–image cross-attention [3]; YOLOv12 introduces attention-centric Area-Attention
blocks [41]; and YOLO-Master introduces sparse MoE expert routing for real-time detection [22]. More broadly,
Switch Transformers and V-MoE establish sparse expert routing as a first-class architectural primitive [5, 39].
This architectural heterogeneity creates a placement challenge that is qualitatively different from Transformer-
centric PEFT: the adapter planner must distinguish dense, grouped, and depthwise convolutions; preserve
DFL and regression-head semantics [20]; accommodate attention, fusion, and expert modules; and produce a
model compatible with checkpointing, adapter merging, ONNX export, and TensorRT deployment [31, 32].
YOLO-PEFT is, to our knowledge, the first systematic framework to formulate PEFT for YOLO-style
detectors asstructure-constrained adapter placement on heterogeneous detector graphsand to translate the
resulting plans into deployable YOLO execution contracts. It differs from prior work along three axes.Unlike
prior PEFT methods, YOLO-PEFT does not introduce a new low-rank parameterisation; it resolves where
any supported parameterisation can be safely applied.Unlike vision adapter methods, it does not assume
a uniform Transformer-block backbone; it explicitly handles the heterogeneous operator and semantic roles
3

--- PAGE 4 ---

present in real-time detectors.Unlike domain-specific detection adaptation studies, it targets the YOLO
family as a broad detector class and enforces operator validity, semantic safety, graph-interface compatibility,
budget feasibility, and deployment constraints as first-class planning objectives. The public YOLO-Master
v26.02 release provides the engineering substrate and deployment-economics benchmark [22]; YOLO-PEFT
contributes the structure-aware planning and execution-contract layer that makes adapter placement valid,
compact, and deployable across this family.
3 Methodology
3.1 Overview and Design Principle
YOLO-PEFT follows one design principle:a detector adapter is useful only if it is structurally valid,
parameter-efficient, and deployable.Unlike language-model PEFT, where target modules are usually
linear projections in a repeated Transformer block, YOLO-family detectors contain heterogeneous operators
and detection-specific heads whose semantics must be preservedbeforeany efficiency gain is meaningful. We
therefore formulate PEFT for YOLO as aconstrained adapter-placement problem: given a detector graph, a
PEFT request, and a resource budget, YOLO-PEFT either returns a deployable adapter plan or refuses the
request when no safe placement exists.
The framework executes four stages in sequence for every request:
1. Parse.GraphParser converts the detector into a role-aware graph, assigning each module anoperator role
and asemantic role(§3.3).
2. Plan.Planner removes structurally invalid and semantically unsafe targets, then assigns adapter ranks
under a parameter budget (§3.4).
3. Install and run.Contract installs the adapters while preserving YOLO training, checkpointing, merging,
and export interfaces (§3.5).
4. Record.Manifest logs the resolved configuration so adapter-only checkpoints can be safely reloaded and
merged later (§3.10).
Architecture-specific guards (§3.6) are injected inside the planning stage for high-risk detector blocks, and
refusal (§3.9) is treated as a first-class output rather than a failure mode. The remainder of this section
defines the problem formally (§3.2), then walks through each stage in the order above.
3.2 Problem Definition
We define YOLO PEFT as the problem ofselecting a subset of detector modules and assigning them adapter
ranks, under structural and deployment constraints, so that the resulting adapter is both trainable and
deployable.
Input.A detector D (equivalently, a directed acyclic graphG = (V, E)where V is the set of detector modules
and E is the set of tensor-flow edges), a user PEFT requestR = (p, T, L,K ), and a deployment budgetB.
Here p is the requested PEFT variant,T is an optional target subset,L is an optional layer interval,K is the
set of allowed ranks, andBis the adapter-parameter budget.
Output.Either an adapter placement planπorREFUSE.
Let Vcand ⊆V denote the modules that remain eligible after request, operator, semantic, and graph-interface
filtering. A valid plan maps these candidate layers to ranks,
π:V cand → {0} ∪ K,
where π(i) = 0means no adapter on layeri and π(i) = r installs a rank-r adapter. The plan must satisfy five
constraints:operator validity,detection-head (semantic) safety,graph-interface compatibility,
parameter-budget feasibility, anddeployment compatibility.REFUSE is emitted when no candidate
plan satisfies these constraints, or when calibrated reliability checks predict a catastrophic placement.
4

--- PAGE 5 ---

Tab. 1.Operator and semantic roles in heterogeneous YOLO graphs. GraphParser assigns operator and semantic roles
by traversing the detector graph. Graph-interface compatibility is enforced conservatively by preserving the original
Ultralytics execution container and by avoiding substitutions that alter tensor shapes or exported module structure.
Role Description / examples
Operator role(what the layer computes)
dense-conv3×3convolution in C2f/C3k2 backbone
grouped-convgrouped conv in lightweight / mobile blocks
depthwise-conv3×3convolution with one group per input channel
linearFC layers in fusion / heads
attentionMHSA in A2C2f / RT-DETR encoder-decoder
fusiontext-image cross-attention (YOLO-World)
expertMoE expert linear / conv (YOLO-Master)
norm/actBatchNorm/SiLU (excluded from placement)
Semantic role(where in the detector)
backboneC2f/C3k2 stages
neckPAFPN fusion path
head-clsdecoupled classification branch
head-regdecoupled regression+DFL branch
head-dfl/dfl-projfixed distribution projection in bbox branch
ov-fusionopen-vocabulary text fusion
moe-routerMoE gate / expert router
3.3 Detector Graph Parser
A modern real-time detectorfθ is a directed acyclic computational graphG = (V, E)whose vertices are
module instances and whose edges are tensor flows. Unlike a Transformer block stack,V is heterogeneous:
the sameConv2dcan play very different roles depending on whether it belongs to the backbone, the neck, the
regression head, or a fixed DFL projection.A target-module rule that is benign for a language model—such
as “target every linear layer”—silently selects DFL projection layers or regression-head sublayers tied to
detection-specific loss geometry when applied to a YOLO detector.
GraphParser therefore assigns each moduletwoindependent roles rather than one:
• Operator roledetermines whether a generic PEFT update preserves the layer’s computational contract.
A depthwise convolution’s per-channel structure is destroyed by a low-rank update that mixes channels;
a grouped convolution’s block-diagonal structure requires group-aware factors; a text-fusion linear has
gradient norms two orders of magnitude above its image-side neighbours and amplifies any unconstrained
update.
• Semantic roledetermines whether the resulting accuracy change is deployment-safe. Adapters in the
regression head can perturb fixed DFL bin projections that the IoU/CIoU loss is specifically calibrated to;
adapters on the MoE router rewire expert assignment and cause silent distribution shift.
Table 1 lists the operator and semantic roles used across the detector families we support. GraphParser
produces the joint(op-role,sem-role)assignment by pattern-matching against a small set of known detector
idioms; an unknown module gets a conservativeunknownrole, which the planner refuses to place adapters
into.
GraphParser also derives anarchitectural fingerprint. LetVrole denote all modules that receive an operator or
semantic role, and letVcand denote candidate modules after request, operator, semantic, and graph filtering.
5

--- PAGE 6 ---

Tab. 2.Paper-calibrated architecture-family profiles used for fingerprint initialisation. Values are derived from the
PASCAL VOC experimental matrix of §4. Unknown families fall back to the improved module-scan path.
Familyϕ attn ϕtext
YOLO-CNN (v8/v9/v10/v11)0.00 0.00
YOLO12 (A2C2f)0.45 0.00
YOLO-World (text-fusion)0.45 0.50
RT-DETR (Transformer)0.85 0.00
YOLO-Master-MoE0.00 0.00
The graphGinduces a bounded10-dim fingerprintϕ(G)∈R 10:
ϕattn(G) = |a ttention|
|Vrole| , ϕ text(G) = |fusion|
|Vrole| ,
ϕmoe(G) = |exper t|
|Vrole| , ϕ dw(G) = |depthwise-conv|
|Vrole| ,
ϕconv(G) = |dense-conv∪grouped-conv|
|Vrole| ,
ϕdepth(G) = |top-level blocks|
30 , ϕ width(G) = log2( ¯C)
10 ,
ϕhead(G) = |θhead|
|θ| , ϕ resid(G) = |residual modules|
|V| ,
ϕnorm(G) = |LayerNorm|
|BN|+|LN|+|GN| .
The original five dimensions distinguish CNN-dominant, attention-heavy, text-fusion, MoE, and depthwise-
heavy graphs without exposing implementation-specific module names. The five extended dimensions (ϕdepth,
ϕwidth, ϕhead, ϕresid, ϕnorm) add scale and structural information that allow the regression model to distinguish,
e.g., , YOLOv8n from YOLOv8x or to detect residual-connection density differences across architecture
families.
The fingerprint is computed in atwo-stagestrategy. First, GraphParser detects the architecture family by
scanning foriconic module types(A2C2f/AAttn for YOLO12, RTDETRDecoder for RT-DETR, text-encoder
fusion for YOLO-World, MoE router/expert for YOLO-Master-MoE) in priority order. For known families,
the base five dimensions are initialised from paper-calibrated profiles (Table 2) to avoid known pathologies
in naive module counting—e.g., , nested AAttn submodules inflatingϕattn beyond1 .0or ResNet backbones
diluting the RT-DETR decoder signal. For unknown or custom architectures, the fingerprint falls back to
animproved module scanthat counts only iconic attention types (rather than every child submodule name)
and computes all ten dimensions from the actual weight topology. This hybrid strategy guarantees reliable
decision-making for the supported families while remaining extensible to novel architectures.
3.4 Structure-Aware Planner
Given the role-aware graph from §3.3, Planner resolves the constrained placement problem of §3.2 in two
ordered stages:it first guarantees validity, then optimizes efficiency.This ordering is intentional:
there is no point optimizing rank allocation over targets that would break the detector’s computational or
deployment contracts. Concretely, the planner (i) filters out invalid operator targets, (ii) filters out unsafe
semantic targets, and only then (iii) assigns budgeted ranks over what remains.
(1) Invalid operator filtering.Depthwise convolutions, normalization/activation layers, and modules with
unknownoperator role are excluded outright:u(i, r) = −∞ whenever op-role(i)is depthwise, norm/act,
or unknown. For grouped convolutions we definer as the total rank budget across groups and allocate a
per-group rank rg withP
g rg = r; using the balanced caserg = r/G, r must be divisible by the group count
G. This is theconvolutional-validityconstraint.
(2) Unsafe semantic filtering.Fixed-bin DFL projections and MoE routers are excluded outright,
u(i, r) = −∞ if sem-role(i) ∈ {head-dfl,moe-router}, as are unsafe regression-head subpaths identified
by the guards in §3.6. This keeps geometry-sensitive head branches and router logits outside the adapter
target set, independent of how much budget is available.
6

--- PAGE 7 ---

(3) Architecture-conditioned hard-policy rules.Before any rank assignment, Planner evaluates a set of
architecture-family-specifichard-policy rulesthat encode empirically observed catastrophic failure modes.
These rules are the primary decision driver; the regression model (step 4) serves as a secondary calibration
layer for edge cases not covered by the rule set. The current rule set comprises:
• RT-DETR refusal:if ϕattn > 0.7and p∈ {LoRA,DoRA,LoHa,LoKr} , emit REFUSE (predicted catas-
trophic collapse).
•YOLO12 DoRA degradation:ifϕ attn >0.3andp=DoRA, degrade to LoRA and cap rank at8.
•YOLO12 high-rank cap:ifϕ attn >0.3and requested rank>8, cap at8with safe-attention mode.
•YOLO-World text-fusion:ifϕ text >0.05andp=LoRA, adapt to LoHa for better text-side stability.
•CNN-only safety:ifϕ attn <0.05, disable attention targets regardless of user request.
A concrete example motivates the necessity of refusal. On PASCAL VOC, RT-DETR-L with LoRA without
the above rule suffers a catastrophic mAP drop from68.8to59 .2( −9.6mAP 50-95), whereas refusal followed
by full fine-tuning preserves the baseline. This single datapoint illustrates that unsafe placement is not merely
suboptimal but actively harmful.
(4) Budgeted rank assignment.Only after steps (1)–(3) have removed unsafe candidates does the planner
optimize efficiency. Given a parsed graphG, a PEFT requestR, a variantp, and a budgetB (in bytes), the
planner forms the surviving candidate setVcand — additionally intersected with any user-specified target
subset T and graph-interface compatibility (avoiding substitutions that change tensor shape or exported
module structure, and preserving the original Ultralytics execution container) — and emits a placement
assignmentπ:V cand → {0} ∪ Kthat solves
max
π
X
i∈Vcand
u(i, π(i);p, ϕ(G))s.t.
X
i
cp(i, π(i))≤B,(1)
where u(i, r; p, ϕ)is theutilityof installing a rank- r adapter on layeri for variantp under fingerprint ϕ, and
the hard constraintP
i cp(i, π(i)) ≤B is theresource-budgetconstraint (typically set as a fraction of the
frozen base checkpoint size, e.g., ,B≈0.15|W0|in our deployment-economics experiments).
For LoRA-style adapters, letb be the byte count per trainable parameter. The formal budget accounting uses
clinear(i, r) =b r(d in +d out), c conv(i, r) =b r(C inkhkw +C out),
and, for grouped convolution,
cgroup(i, r) =b
GX
g=1
rg((Cin/G)khkw +C out/G).
Inrule-only mode,uis a deterministic formal priority score:
u(i, r;p, ϕ) =u op(i) +usem(i) +urange(i) +urank(r)−λc p(i, r).
Candidate pairs with non-positive utility are discarded before placement.
(5) Regression calibration and LOVO validation.The planner additionally maintains a linear regression
model
∆mAP≈β 0 +β 1ϕattn +β 2ϕtext +β 3ϕdw +β 4ξp,(2)
where ξp is the variant-level coefficient (fitted via least squares on canonical data points). Default coefficients
(β0, β1, β2, β3, β4) = (0.0656, 0.0026, 0.0, 0.0054, 1.0)are calibrated on our PASCAL VOC experimental matrix
(§4).
This regression servestwopurposes: (i) after a hard-policy decision is made, it fills thepredicted_delta
field of the decision record for audit and reproducibility; (ii) incalibrated mode, it re-evaluates cases that
7

--- PAGE 8 ---

fall outside the coverage of the hard-rule set—e.g., , a novel architecture family or a newly released PEFT
variant—and may trigger an additionalREFUSE when the predicted∆mAP falls below a catastrophe threshold.
Unless otherwise stated, planner ablations use rule-only mode; calibrated mode is evaluated only under
leave-one-variant-out (LOVO) cross-validation or on architectures not used to fit the corresponding predictor.
LOVO accuracy, precision, recall, andF1 are reported in Table??of §4.
Deployment compatibility is enforced after planning by Contract (§3.5), which may refuse plans that cannot
be lowered into a mergeable or export-compatible runtime.
3.5 YOLO-Compatible Runtime Contract
A PEFT plan that improves validation accuracy but cannot be merged or exported is not a
deployable solution.Unlike a generic module wrapper, Contract must remain compatible with the full
Ultralytics deployment surface: the forward graph, checkpoint format, convolution fusion, ONNX/TensorRT
export, adapter merging, and adapter-only save/load. Contract translates each placement entry(i, r)into an
installation procedure that preserves checkpoint compatibility through a name-stable save/load shim and
enforces four deployment invariants:
I1.Base checkpoint loading remains unchanged.
I2.Adapter checkpoints contain only adapter tensors and runtime metadata.
I3.Merged models restore ordinary YOLO module structure.
I4.Export tools see an export-compatible YOLO model.
YOLO-PEFT exposes one user surface and dispatches to two runtime families. The PEFT backend [28] is
the default path for LoRA, RS-LoRA, DoRA, LoHa, LoKr, AdaLoRA, IA3, OFT, BOFT, and HRA. The
in-repo fallback backend is intentionally narrower: it provides manual convolutional LoRA wrapping when
PEFT is unavailable, explicitly bypassed, or a fallback path is requested. Both backends share the same
target-resolution and lifecycle contract, including adapter-only saving, loading, merging, and export-safe
restoration. Full backend routing details are given in Appendix B.
Proposition 1(Merge equivalence for fallback convolutional LoRA).For plain LoRA adapters installed
through the in-repo fallback convolutional backend, the adapter branch admits an exact deploy-time merge
into the host convolution. For each groupg, the update is∆Wg = (BgA⊤
g )reshaped to the corresponding
convolutional kernel. After W0 ←W 0 + s∆W, the wrapper can be removed and the resulting host convolution
produces the same output up to numerical tolerance. The proof is given in Appendix A.
Proposition 1 only applies to the fallback plain convolutional LoRA path. Other variants are routed by
runtime responsibility rather than by a broad theorem about their mathematical mergeability (Table 3).
Ultralytics-style deployment requires special care because validation and export may fuse convolutional blocks
before adapters have been merged. The contract layer therefore guards fusion during adapter training and
validation, then removes the wrapper after merge so that export tools again see an ordinary YOLO module
structure. Unmerged wrappers are not exported directly; export is allowed only after a successful merge or
through a PEFT-managed export path.
The user-facing surface remains the existing training call plus three adapter lifecycle operations: saving
adapter-only checkpoints, loading them onto a fresh base checkpoint with metadata checks, and merging them
back into an export-compatible detector. The GraphParser, Planner, Contract, and Manifest components are
wired internally before optimiser construction, so no new trainer abstraction is exposed to the practitioner.
3.6 Architecture-Specific Safety Guards
A planner covering heterogeneous detector families needs guard rails against architectures that break LM-style
PEFT defaults. YOLO-PEFT implements three conservative architecture-specific safety guards inside Planner,
combining semantic-validity filters with training-stability overrides. These guards are motivated by the failure
modes analysed in §4.
8

--- PAGE 9 ---

Tab. 3.Runtime support matrix for adapter variants. PEFT is the default backend for supported variants, while the
in-repo fallback backend is intentionally narrower and covers manual convolutional LoRA wrapping. Proposition 1
applies only to the fallback convolutional LoRA path; PEFT-managed variants delegate their parameterisation and
merge behavior to the PEFT runtime.
Variant / strategy Default backend Fallback support Merge / export path
LoRA PEFT Yes, convolution only PEFT merge; fallback uses Prop. 1
RS-LoRA PEFT No PEFT-managed merge
DoRA PEFT No PEFT-managed merge
LoHa PEFT No PEFT-managed merge/export if supported
LoKr PEFT No PEFT-managed merge/export if supported
AdaLoRA PEFT No PEFT-managed merge
IA3 PEFT No PEFT-managed merge/export if supported
OFT PEFT No PEFT-managed
BOFT PEFT No PEFT-managed
HRA PEFT No PEFT-managed
LoRA strategies selected LoRA backend LoRA only follow selected LoRA backend
Few-shot LoRA fallback / specialized Yes fallback merge after rank-mask resolution
MoLoRA fallback / specialized Yes expert-merge after top-kmask resolution
TheYOLO12 Area-Attention guardapplies when GraphParser detects theA2C2f / ABlock / AAttn
pattern (YOLO12’s Area-Attention); the placement exclusion dropsattn.{qkv,proj,pe} and the internal
MLP convolutions from the placement set. The associated training safeguard forcesα-warmup to ≥3epochs
and caps the LoRA learning-rate multiplier to1.0.
TheRT-DETR MSDeformAttn guardexcludessampling_offsets(whose grid-initialised bias encodes
the deformable sampling grid) andattention_weights (whose softmax is zero-init and saturates under any
LoRA delta) from the placement set. These layers are excluded by default; adapting them would require an
explicit code-level override and low-rank / long-warmup tuning.
TheDFL projection guardtreats the1 ×1projection’s weight as the fixed bin0, 1, . . . , n−1projector that
the IoU/CIoU/DFL loss is calibrated to. The framework matches it by name (*.dfl.*) and unconditionally
excludes it from placement.
3.7 Training Strategies
Beyond adapter placement, YOLO-PEFT provides four complementary training strategies that are applied
automatically when the planner resolves to an ADAPT or ACCEPT decision. All strategies are optional
(controlled viaLoRAConfig) and are implemented in theLoraTrainingStrategyclass.
Strategy 1: Layer-wise LR decay.The learning rate of adapter parameters decays exponentially with
backbone depth: ηℓ = η0 ·ρ dℓ, where dℓ ∈ [0, 1]is the normalised layer depth and ρ = 0.85is the decay
rate. To avoid creating one optimizer param-group per layer (which slows the optimizer for YOLO’s∼23
top-level blocks), factors are rounded to one decimal place, producing∼3–5 LR groupsin practice. This is
an implementation-level grouping that preserves the stratification intent while remaining optimizer-efficient.
Strategy 2: Alpha warmup.LoRA scaling α/r is ramped from0to its target value over the firstN
epochs via a cosine ease-in curve. This prevents initial training instability on attention-heavy architectures
(especially YOLO12) where a full-strength adapter at epoch0can cause loss divergence. The implementation
supports five PEFT-version compatibility paths (dict-styleα, direct numeric attribute, property-based, scaling
attribute, and config-dict fallback) to cover PEFT0.13through0.18+.
Strategy 3: Orthogonal regularization.Every N batches (default N = 10), an auxiliary loss
λortho(∥A⊤A−I∥ F + ∥B⊤B−I∥ F )is added to the training objective, with λortho = 0 .5in our default
configuration. A chunked computation path (chunk size32) bounds peak GPU memory toO(32 ×r 2)instead
ofO(L lora ×r 2).
Strategy 4: Dynamic dropout scheduling.The LoRA dropout rate is linearly interpolated from a start
value (default0 .0) to an end value (default0.15) beginning at a specified fraction of total epochs (default
9

--- PAGE 10 ---

30%). This provides low dropout in early phases (preserving gradient signal) and higher regularisation later
(preventing overfitting).
3.8 Extended Adapter Families: MoLoRA and Few-Shot LoRA
YOLO-PEFT supports two additional adapter families beyond the standard PEFT variants, both implemented
in the in-repo fallback backend.
MoLoRA (Mixture-of-LoRA).For detectors that already contain Mixture-of-Experts (MoE) routing
(YOLO-Master), YOLO-PEFT provides a Mixture-of-LoRA layer that replaces each expert’s dense update
with a sparse set of rank-r LoRA experts. A Top-k router (with Linear, Spatial, or Hybrid routing modes)
selects which LoRA experts are active per sample or per spatial location. The framework supports balance loss,
Z-loss (router-logit stability), and diversity loss for load balancing; domain pre-allocation, continual-learning
expert freeze/unfreeze, and EMA teacher distillation are also implemented.As the present work focuses on
standard PEFT variants evaluated on PASCAL VOC, quantitative experimental results for MoLoRA are
deferred to future work.Full architectural details are given in Appendix??.
Few-Shot LoRA.When the training set is small (fewer than500images or fewer than5samples per class),
YOLO-PEFT can activate a few-shot mode that augments the standard LoRA path with DropConnect
regularisation, knowledge distillation from a teacher model (optional), hierarchical multi-layer distillation,
variational rank selection, and curriculum sampling. An EMA teacher with cosine-scheduled distillation weight
provides progressive self-distillation.
Importantly, on small datasets such as PASCAL VOC, PEFT canoutperformfull fine-tuning by preserving
backbone pretraining knowledge while adapting task-specific layers. For example, YOLO12s + HRA achieves
74.5mAP 50-95 on PASCAL VOC, compared to66.6for full fine-tuning—a+7.9point gain. At the methodology
level, this phenomenon is jointly guaranteed by semantic-role filtering and operator-validity constraints: the
backbone remains frozen while only safety-verified adapter parameters are updated, preserving pretrained
representations that would otherwise be overwritten by low-data full fine-tuning. See Appendix?? for
configuration details.
3.9 Refusal and Algorithm Summary
A planner that always returns a placement is dangerous when no placement satisfies the constraints.When
no placement satisfies the operator, semantic, graph, budget, or reliability constraints defined
above, YOLO-PEFT returnsREFUSE rather than producing an unsafe adapter.Planner treats
refusal as a first-class output, with a structured explanation listing the binding constraint: operator, semantic,
graph, budget, deployment, or reliability.
Algorithm 1 summarizes the full placement pipeline, combining the four stages of §3.1 with the refusal logic
above. Each numbered step maps to a subsection: parsing (§3.3), operator and semantic filtering (§3.4),
hard-policy rules (§3.4), budgeted rank assignment (§3.4), regression calibration, and contract lowering (§3.5).
10

--- PAGE 11 ---

Algorithm 1YOLO-PEFT adapter planning.
Input:DetectorD, PEFT requestR= (p, T, L,K), budgetB, fingerprintϕ(G)
Output:Deployable placementπorREFUSE
1:ParseDinto role-aware graphG= (V, E); assign operator role and semantic role to everyv∈V ▷§3.3
2:Computeϕ(G)via two-stage fingerprint (family detection + improved module scan)▷§3.3
3:V cand ←EnumerateConvLinear(V)
4:V cand ←ApplyLayerKernelChannelFilters(Vcand, R)
5:V cand ←ApplyGroupedDepthwiseFeasibility(Vcand, R)▷operator validity
6:V cand ←ApplySemanticExclusions(Vcand, G)▷semantic safety
7:V cand ←IntersectUserTargets(Vcand, T)
8:ifV cand =∅then
9:returnREFUSE▷no structurally valid target
10:end if
11:Evaluate hard-policy rules; emitADAPTorREFUSEif triggered▷§3.4, step 3
12:Assign ranks: solve Eq. (1) underKand budgetB; discard candidates withu≤0▷budgeted rank
assignment
13:ifπ=0or P
i cp(i, π(i))> Bthen
14:returnREFUSE▷budget infeasible
15:end if
16:Fillpredicted_deltavia Eq. (2)
17:ifcalibrated modeandpredicted∆mAP below catastrophe thresholdthen
18:returnREFUSE▷reliability
19:end if
20:Lowerπinto YOLO runtime contract via Contract▷§3.5
21:returndeployable adapter planπ
3.10 Runtime Metadata and Compatibility Checking
A configuration-only framework invites afleetof per-customer adapters. The current system persists runtime
metadataratherthanafullcryptographicmanifest. Itrecordsthebackend, variant, targetmodules, freeze/head
settings, and backend-specific metadata such as the fallback weight file. A full base-checkpoint, model-YAML,
class-hash, or architecture-fingerprint manifest is a deploy-time extension rather than a current hard-checking
requirement. The metadata fields are summarized in Appendix C.
4 Experiments
The experiments are organized around four empirical questions rather than around a catalogue of tables: (Q1)
does YOLO-PEFT preserve the deployment economics of adapter-only training, saving, loading, merge, and
export; (Q2) is PEFT behavior architecture-conditioned rather than governed by a universal variant ranking;
(Q3) can the planner refuse catastrophic placements on high-risk detectors; and (Q4) does structure-aware
placement improve over LM-inherited PEFT rankings. The FewShotLoRA evaluation protocol is provided
in Appendix G; we do not claim empirical few-shot gains until the corresponding runs are completed. Each
of the following subsections follows the same analytic pattern: we state the question, point to the evidence,
explain the mechanism behind the observed pattern, and close with an explicit Finding that the rest of the
paper can reference.
4.1 Setup
We adopt PASCAL VOC07+12 [4] trainval as the primary diagnosis dataset and report on val2007.1 All runs
are released as a singlewandb_export.csv table together with this submission. The FewShotLoRA protocol
is specified in Appendix G; we do not report few-shot numbers until the corresponding VOC1/2/5/10-shot
runs are completed. The core W&B table reports YOLO11s [17] (dense-convbaseline, ∼9.4M params,21 .6
GFLOPs), YOLO12s [41] (dense-conv+attention,∼9.3M params,21 .6GFLOPs), and RT-DETR-l [27]
1VOC trainval (∼16k images,20classes) is a controlled mid-scale corpus that exposes operator-level PEFT failures while
remaining tractable on a single A100 (roughly5–8hours per300-epoch run).
11

--- PAGE 12 ---

n s m l x
Scale
0
20
40
60
80
100
120Size (MB)
6 2.1
19
4.1
41
6.6
51
9.4
115
14.1
YOLO11 series
Base model
LoRA adapter
n s m l x
Scale
0
20
40
60
80
100
120Size (MB)
6 2.3
19
4.3
41
6.8
54
9.8
119
14.7
YOLO12 series
Base model
LoRA adapter
On-disk size: full model vs LoRA adapter (data: YOLO-Master v26.02 release)
Fig. 1.On-disk size: full model vs. LoRA adapter.Numbers from the YOLO-Master v2026.02 release [22],
end-to-end reproduced by our contract-only implementation. The adapter-to-base ratio shrinks from∼37%at scale n
to∼12%at scale x.
(pureattentionencoder+decoder, ∼32.8M params,108 .1GFLOPs). The extended architecture-conditioned
matrix further includes YOLOv8n, YOLO11n, YOLO12n, and YOLO-World-s to cover CNN-only, attention,
text-fusion, and Transformer-detector regimes. Thus Table 4 uses the s-scale core W&B runs, while Fig. 4 uses
the smaller n-scale variants at320resolution for the broader diagnostic sweep. The core W&B runs in Table 4
use300epochs, image size640, and batch256for PEFT runs (YOLO12s reduces to128for RT-DETR-l
due to its memory footprint). The extended diagnostic matrix in Fig. 4 uses the same300-epoch protocol at
image size320to make the larger variant–architecture sweep tractable. Optimisation uses AdamW [26] with
lr0=0.01, lrf=0.01, weight-decay5 × 10−4, momentum0 .937, cosine schedule, warmup3epochs, mosaic close
at epoch290. PEFT-specific defaults: rank r=16(with a separate r={8, 32} sweep on YOLO12s, Tab. 14),
α=2r=32, dropout0 .05, orthogonality regulariser weight0.5, ortho-frequency every10steps, RS-LoRA scaling
on by default, DoRA’s magnitude branch off by default; details in Appendix H. All runs use4×NVIDIA
A100 (80GB) under DDP, CUDA12.1, PyTorch2.3.1, and HuggingFace PEFT0.18.1. Reported metrics
are final-epoch validation readings on VOC val2007 logged to W&B (projectYOLO-Master-Exp); the exact
run-time ranges are∼2.7ks for Full-SFT to∼69ks for the longest HRA-on-YOLO11s run.
To avoid overstating the deployment-economics reproduction as a novel contribution, we treat §4.2 as a validity
check rather than a core result: the paper’s central claims are the structure-aware placement framework (§3)
and the architecture-conditioned failure analysis (§4.3–§4.7).
4.2 Deployment economics: a validity check, not the core contribution
Question.Before analyzing accuracy and refusal behavior, we first ask a prerequisite question: does YOLO-
PEFT’s adapter lifecycle (train, save, load, merge, export) actually preserve the deployment-time economics
that motivate PEFT in the first place? If it did not, the accuracy results in later subsections would be moot.
The numbers in this section are not new contributions; they areindependent reproductionsof the public v26.02
release [22], executed entirely under Contract’s dual-backend contract. The PEFT backend is the default
route for supported variants, while the fallback path covers convolutional LoRA adapters when requested or
required.
Evidence.Figure 1 reproduces the public release’s compression-ratio table end-to-end for both YOLO11
and YOLO12 at all five scales. The peak is YOLO11x at8.13× (114.6→14.1MB). Figure 2 reports peak
training memory across all ten scales (YOLO11/12× n/s/m/l/x). Compared to Full-SFT, the contract-only
LoRA run uses5.6GB instead of21 .4GB (YOLO11s, batch32, FP16), a74%cut in line with the release.
For ten downstream YOLO11x adapters, module substitution requires10×114.6=1,146MB; YOLO-PEFT
ships114 .6 + 10×14.1 = 255.6MB, a77 .7%saving for the fleet and87.7%saving for distributing each new
adapter. Figure 3 reproduces the release benchmark of three PEFT variants against Full-SFT. Atr=16,
LoRA preserves95 .7%of full-SFT mAP@0.5:0.95(0 .656vs.0 .685); the gap is smaller than our measured
12

--- PAGE 13 ---

Fig. 2.Training-time GPU memory.YOLO11 and YOLO12 under LoRA withr=8, gradient checkpointing on.
Our contract-based backend matches the public release’s70–75%reduction on every scale.
Fig. 3.PEFT methods vs. Full-SFT on YOLO.Public-release benchmark of LoRA, DoRA, LoHa against
Full-SFT on YOLOv11-s, COCO,300epochs.Top-line message: LoRA atr=16retains95.7%of full-SFT
mAP@0.5:0.95, with the adapter being one fifth of the full model.
cross-seed standard deviation (σ=0.033).
Mechanism.The storage benefit is fleet-level rather than single-model-level: for a single task, an adapter
package is not necessarily cheaper than one full model, since the frozen base checkpoint still has to be shipped
once. The economics only appear once a base detector is reused across multiple downstream variants — the
shared-base design amortizes the one-time cost of the frozen backbone, and each additional adapter is nearly
free by comparison. This is why the87.7%per-adapter saving above grows with fleet sizeK rather than being
a fixed property of any single adapter (see Appendix G/App. E for theK=1regime, where savings can be
small or negative — we discuss this limitation explicitly in §??).
Finding 1.YOLO-PEFT’s deployment advantage comes from amortizing one frozen detector across many
task-specific adapters, not from any single adapter being intrinsically cheap; the benefit is a property of the
fleet, not of an individual model.
We next treat variant–architecture compatibility as a measurable quantity, not a hyperparameter to be
searched. The remaining experiments answer Q2–Q4 with a W&B-grounded matrix, refusal stress tests,
architecture-family validation, predictive validation, and an LM-inherited ranking comparison.
Core W&B measured matrix.Table 4 reports the core W&B-grounded s-scale runs used throughout the
analysis. Unlike Fig. 4, which visualizes a broader n-scale variant–architecture sweep, this table shows the
concrete training configurations, placement targets, mAP, parameter cost, and GFLOPs. Appendix D retains
the companion numerical matrices behind Fig. 4.
13

--- PAGE 14 ---

Tab. 4.PEFT-on-YOLO measured matrix(W&B export, VOC val2007,300epochs,640resolution). Anchors are
Full-SFT runs of the same backbone. “rs” = RS-LoRA scaling, “DoRA” = magnitude branch, andop-targetsis the
set of placement-eligible operator roles selected by the planner.Boldnumbers denote variants that match or exceed
Full-SFT.
Backbone Variantr αrs DoRAop-t argetsmAP 50 mAP 50:95 ∆Trainable GFLOPs
YOLO11s(dense-convonly,ϕ attn =0, anchor0.6428)
YOLO11s Full-SFT (anchor) – – – – all 0.8344 0.6428 — 9.44M 21.6
YOLO11s LoRA16 32true false conv0.88650.7138+0.071010.44M25.7
YOLO11s DoRA16 32true true conv0.88650.7138+0.071010.44M25.7
YOLO11s DoRA (no-rs ablation)16 32false true conv0.8350 0.6479 +0.0050 10.46M25.7
YOLO11s LoHa16 32true false conv0.8620 0.6788 +0.0359 11.45M21.6
YOLO11s LoKr16 32true false conv0.87500.7033+0.06059.51M21.6
YOLO11s IA 3 – – – – conv0.87170.6980+0.05529.45M21.6
YOLO11s HRA16 32true false conv0.90230.7276+0.084810.24M0.46
YOLO12s(dense-conv+a ttention,ϕ attn ≈0.45, anchor0.6662)
YOLO12s Full-SFT (anchor) – – – – all 0.8532 0.6662 — 9.26M 21.6
YOLO12s LoRA (r=8)8 16true false conv+attn0.89740.7288+0.06269.66M23.6
YOLO12s LoRA (r=16)16 32true false conv+attn0.90070.7307+0.064510.06M25.6
YOLO12s LoRA (r=32)32 64true false conv+attn0.90220.7363+0.070110.85M29.6
YOLO12s LoRA (no-rs ablation)16 32false false conv+attn0.88530.7094+0.043210.07M25.6
YOLO12s LoRA+DoRA, no rs16 32false true conv+attn0.8041 0.6112−0.0550 10.07M25.6
YOLO12s LoHa16 32true false conv+attn0.89070.7222+0.056010.85M21.6
YOLO12s IA 3 – – – – conv+attn0.89280.7210+0.05489.27M21.6
YOLO12s HRA16 32true false conv+attn0.91110.7453+0.07919.91M3.84
RT-DETR-l(purea ttention,ϕ attn ≈0.85, anchor0.6833)
RT-DETR-l Full-SFT (anchor) – – – – all 0.8614 0.6833 — 32.85M 108.1
RT-DETR-l LoRA-class refused ‡ – – – – – — — — (refused) — —
‡The structure-aware planner emitsREFUSEon RT-DETR-l: under Eq. (3) every LoRA-family variant predicts∆<−0.05,
agreeing with the unstable training trajectory we observe within the first∼15epochs of any LoRA configuration on this backbone.
We also note that several PEFT configurations in Table 4 outperform Full-SFT by6–8mAP points. We
attribute this primarily to the regularization effect of freezing most of a pretrained detector under a mid-scale
dataset (VOC), where full fine-tuning of all parameters can over-adapt; freezing the backbone and constraining
updates to a low-rank subspace acts as an implicit regularizer under this data budget. To avoid overstating
this effect, we report both absolute mAP and retention relative to Full-SFT throughout, and we cross-validate
Full-SFT and the strongest PEFT configuration across three seeds (cross-seedσ≤ 0.006mAP on the matrix
cells, and σ=0.033specifically for the LoRA-vs-Full-SFT gap in Fig. 3), confirming the Full-SFT baseline is
not simply under-tuned relative to the PEFT runs.
4.3 Architecture-conditioned PEFT behavior
Question.Is there a single PEFT variant, or a single stability ranking of variants, that works best across
detector architectures — the way LM-oriented PEFT rankings are often treated as near-universal — or is
stability conditioned on the detector’s structure?
Evidence.Figure 4 visualizes the broader14-variant × 5-architecture matrix. The catastrophe rate
(∆ <− 0.05) climbs monotonically withϕattn:1 /14on YOLOv8n,0 /10on YOLO11n,6 /7on YOLO12n,4 /7
on YOLO-World-s, and7/7on RT-DETR-l. The heatmap shows that PEFT variants do not preserve their
relative ordering across detector families: methods that are stable on CNN-only YOLO variants can collapse
on attention or text-fusion detectors, while LoHa remains comparatively robust in the non-Transformer regime
but still fails on RT-DETR-l. The pairwise Kendall-τ analysis (Fig. 6b) further confirms that rankings are
consistent mainly within closely related CNN architectures and become weak or negative across architecture
families.
Mechanism.Catastrophe rate is not driven by which variant is used per se, but by how much a given
detector’s execution graph resembles the Transformer-linear-projection structure that most PEFT methods
(LoRA, DoRA, LoHa, LoKr, etc.) were originally designed and stability-tuned for. Asϕattn rises — i.e.,
as the graph shifts from dense/grouped/depthwise convolutions toward attention and cross-modal fusion —
the implicit assumptions behind these methods’ stability defaults (e.g., additive low-rank updates on linear
14

--- PAGE 15 ---

YOLOv8n
(conv)
YOLO11n
(conv)
YOLO12n
(conv+attn)
YOLO-World-s
(conv+lin+attn)
Full-SFT
LoRA-vanilla
LoRA+ -warmup
LoRA+ortho
LoRA+layer-decay
LoRA+ortho+decay
LoRA+spectral
LoRA+ortho+spec
DoRA
LoHa
LoKr
IA³
HRA
BOFT
AdaLoRA
0.000 0.000 0.000 0.000
+0.008 +0.011 -0.086 -0.097
+0.020 +0.017
+0.019 +0.021 -0.087 -0.002
+0.022 +0.022
+0.021 +0.020
-0.009 -0.007 -0.084 -0.033
-0.010 -0.007 -0.088 -0.050
+0.006 +0.007 -0.081 -0.292
+0.009 +0.012 +0.031 +0.081
+0.008 -0.204 -0.056 -0.167
+0.007
+0.002
-0.069
0.000
 mAP@0.5:0.95 vs Full-SFT    42-cell PEFT × Architecture matrix
0.10
0.05
0.00
0.05
0.10
 mAP
Fig. 4.Architecture-conditioned PEFT behavior.∆mAP@0.5:0.95vs Full-SFT for14PEFT variants on5
heterogeneous detection backbones (VOC,300ep, img320, seed 0). Bright red cells are catastrophic failures
(∆<−0.05, i.e., more than5mAP points); for collapsed RT-DETR-l runs,∆ =−0.600denotes the clipped failure
value used for visualization. The catastrophe rate climbs monotonically withϕattn:1/14on YOLOv8n,0/10on
YOLO11n,6 /7on YOLO12n,4 /7on YOLO-World-s, and7 /7on RT-DETR-l. Bold rows (LoRA+layer-decay, LoHa)
are relatively stable planner-selected configurations; no universal PEFT ranking survives across CNN, attention,
text-fusion, and Transformer detectors.
projections with roughly homogeneous gradient scale) increasingly fail to hold.
Finding 2.PEFT reliability on YOLO is architecture-conditioned rather than governed by a universal
variant ranking: stability rankings inherited from language-model PEFT do not transfer to detectors, and
adapter selection must be conditioned on the target detector’s architecture, not solely on the PEFT method’s
LM-side reputation.
4.4 Failure modes and refusal under high-risk detectors
Question.Given that failures are architecture-conditioned rather than random, what specific mechanisms
drive them, and — critically — should a PEFT framework always returnsomeadapter, or is refusing to place
an adapter sometimes the correct decision?
Evidence and mechanism (variant-level failures). F1. Variant collapse is architecture-specific.
DoRA loses29.2mAP on YOLO-World yetgains0 .7on YOLO11n; LoKr loses20.5on YOLO11n yetgains0 .8
on YOLOv8n. The relative ordering is unstable across architectures, consistent with Finding 2.F2. LoHa is
15

--- PAGE 16 ---

−0.7 −0.6 −0.5 −0.4 −0.3 −0.2 −0.1 0.0
Δ mAP@0.5 : 0.95 vs.\ Full-SFT (RT-DETR-l, ϕattn = 0.85)
LoRA-baseline
DORA
LOHA
LOKR
LoRA+ortho+spectral
LoRA+ortho-reg
LoRA+spectral
-0.229
-0.600
-0.600
-0.600
-0.600
-0.600
-0.600
Every PEFT variant collapses on a Transformer detector (7/7 catastrophic)
catastrophic threshold (−0.05)
Fig. 5.Refusal prevents catastrophic detector adaptation.On RT-DETR-l, every PEFT variant that survives
training falls below the catastrophic threshold; the planner refuses and falls back to Full-SFT.
Tab. 5.Planner-selected adaptation across detector families on PASCAL VOC. CNN rows aggregate
YOLOv3/v5/v6/v8/v9/v10/YOLO11; the full eleven-backbone coverage table is retained in Appendix E.
‡RT-DETR-l: the plannerrefusesall14variants.
Architecture class Backbones Decision Adapter Status
CNN YOLOv3/v5/v6/v8/v9/v10/YOLO11 LoRA+decay 1.5–4.0MB accept,+0.018–+0.022
CNN+attention YOLO12 LoHa 2.5MB accept,+0.031
Text-fusion YOLO-World LoHa 2.1MB accept,+0.081
Transformer detector‡ RT-DETRREFUSE– avoid−0.229
MoE YOLO-Master LoRA(exp) 3.4MB accept,+0.025
uniquely robust within the non-Transformer regime.LoHa is the only variant strictly beating Full-SFT
on all four non-Transformer backbones (avg.+3.3mAP). Once ϕattn exceeds ∼ 0.5, even LoHa collapses.
F3. Regulariser stacking is non-additive.Spectral regularization on top of LoRAloses0.9mAP on
YOLOv8n; spectral+ortho loses1 .0. Layer-decay alone is the strongest single addition, indicating that
training-side priors interact rather than compose linearly.F4. DoRA and text-fusion fail mechanistically,
not just empirically.The image and text branches of YOLO-World have order-of-magnitude different
gradient norms; DoRA’s magnitude vector amplifies this mismatch and diverges within∼15epochs. This is
a named mechanism — gradient-norm mismatch between modality branches — that has no counterpart in
uni-modal LMs, which is precisely why an LM-tuned method like DoRA cannot be assumed safe here.F5.
No universal best configuration.The argmax over variants is architecture-dependent, reinforcing Finding
2 at the level of individual failure mechanisms rather than aggregate statistics.
Evidence and mechanism (refusal on RT-DETR-l).RT-DETR-l provides a stress test for whether a
PEFT framework should always return an adapter. Whereϕattn ≥0.5,everyof the seven swept variants falls
below the catastrophic threshold (Fig. 5), indicating that the failure is systematic — rooted in RT-DETR-l’s
deformable-attention structure — rather than a hyperparameter accident that a different learning rate or
warmup schedule could fix. In this regime, returning even the least-bad swept adapter would still produce an
unusable detector. YOLO-PEFT’s planner therefore fires its refusal mechanism and falls back to Full-SFT
rather than silently shipping a degraded model — refusal here is not a limitation of the method, but a
deliberate and necessary design decision.
The planner converts the worst rows of Fig. 4 (DoRA on YOLO-World, LoRA on YOLO12n:−0.10, −0.09) into
positive deltas in Table 5 by refusing DoRA/LoKr and defaulting to LoHa under the operator/semantic/budget
16

--- PAGE 17 ---

Tab. 6.Predictive validation summary for Eq.(3). The main quantity is leave-one-variant-out catastrophe prediction
on unseen variants (∆<−0.05).
Metric Value
R2 on fitted matrix0.762
lovocatastrophe accuracy86.7%
lovocatastrophe recall0.944
lovocatastropheF 1 0.850
−0.6 −0.5 −0.4 −0.3 −0.2 −0.1 0.0 0.1
true Δ mAP@0.5 : 0.95
−0.6
−0.5
−0.4
−0.3
−0.2
−0.1
0.0
0.1
predicted Δ from scaling law
(a) Scaling-law fit: R2 = 0.762 on n = 45
YOLOv8n (n=14)
YOLO11n (n=10)
YOLO12n (n=7)
YOLO-World-s (n=7)
RT-DETR-l (n=7)
v8n 11n 12n W-s
RT-DETR
v8n
11n
12n
W-s
RT-DETR
+1.00 +0.69 +0.33 +0.24 +0.00
+0.69 +1.00 -0.14 +0.33 +0.18
+0.33 -0.14 +1.00 -0.05 -0.18
+0.24 +0.33 -0.05 +1.00 -0.18
+0.00 +0.18 -0.18 -0.18 +1.00
(b) Pairwise Kendall's τ: rank-reversal evidence
−1.00
−0.75
−0.50
−0.25
0.00
0.25
0.50
0.75
1.00
Kendall's τ
Fig. 6.Vision PEFT Scaling Law fit and pairwiseτ. (a)Predicted vs. true∆mAP for all45cells; the dashed
diagonal is the identity, the red dotted lines mark the catastrophe threshold∆=−0.05.(b)Pairwise Kendall’sτ
heatmap.
constraints. The full eleven-backbone coverage table is reported in Appendix E; Table 5 aggregates it by
architecture class to avoid over-representing closely related CNN-family detectors. On RT-DETR-l the planner
fires its hardest decision: it refusesall14variants, returning the practitioner to Full-SFT with a one-line
explanation.
Finding 3.Refusal is necessary for detector PEFT, not a fallback for missing coverage: some architecture–
adapter combinations (e.g., any swept LoRA-family variant on RT-DETR-l) are systematically unsafe, and
YOLO-PEFT treats refusing to place an adapter as a first-class, valid planning decision rather than a failure
mode.
4.5 Predictive validation: Leave-one-variant-out
Question.Finding 2 shows that architecture conditions PEFT stability post hoc. Can this relationship be
captured by a compact, predictive model that generalizes to a PEFT variant the model has never seen — i.e.,
is this a reusable law rather than a fit to the observed cells?
Evidence.A scaling law that fits the training cells but cannot predictunseenvariants is not a law. We hold
out, in turn, each variantp∗ from the matrix, refit
∆mAP(p, G)≈β 0 +β 1ϕattn(G) +β 2ϕtext(G) +β 3ϕdw(G) +β 4ξp,(3)
on the remaining13variants, and predict∆( p∗, G)for all5architectures. The regression is fit on observed
non-missing cells only; missing variant–architecture cells are excluded rather than imputed.
Mechanism.The fingerprint ϕ(G)captures exactly the architectural axes (attention ratio, text-fusion ratio,
depthwise ratio) that Finding 2 identified as driving instability, so a regression overϕ generalizes across
variants even though it was never shown examples of the held-out variant’s behavior — the model is predicting
fromarchitecture, not from memorized per-variant statistics.
Finding 4 (predictive).Catastrophic PEFT failure on a new, unseen variant can be predicted from a
5-dimensional architectural fingerprint with86.7%leave-one-variant-out accuracy (F1=0.850), before any
training run — this is what makes calibrated-mode refusal (§3.4) actionable in practice rather than only
17

--- PAGE 18 ---

Tab. 7.Compact ablation suite. Each row compresses one detailed appendix ablation into the design question it
answers.
Design question Key result Conclusion
Contract necessity Naive substitution and Contract reach the same mAP
(0.7138); only Contract passes ONNX/TRT/Ckpt
Runtime contract is required
Planner constraints0.6900→0.7094→0.7307Semantic filtering is the main
stability driver (Finding 5)
Rank choice0.7288/0.7307/0.7363r=16is a Pareto default, not the
accuracy optimum (Finding 6)
Variant dependence HRA/LoRA/LoHa ordering shifts No universal PEFT ranking
(Finding 2)
RS-LoRA×DoRA DoRA without RS collapses to0.6112Training priors interact
non-additively
Adapter fleet1146.0MB→255.6MB Fleet-level saving of77.7%
(Finding 1)
diagnostic in hindsight. A practitioner facing anewvariant or a new architecture can computeϕin milliseconds,
plug into Eq.(3), and predict failure before training; in calibrated mode, the planner uses this prediction to
reject high-risk placements, evaluated here strictly under leave-one-variant-out prediction.
4.6 Planner ablation: which constraint actually buys stability
Question.The planner in §3.4 applies constraints in a specific order (operator validity, then semantic safety,
then budgeted ranks). Is this ordering — and are these constraints — actually load-bearing, or are they
conservative engineering choices that could be dropped without much cost?
Evidence.On YOLO12s, removing all planner constraints and placing adapters naively gives only0.6900
mAP50:95. Adding operator validity (excluding depthwise/norm/unknown targets) improves the result to
0.7094. Adding detection-head semantic exclusions (DFL, MoE router, unsafe regression subpaths) further
improves the result to0.7307— matching the planner’s final selection in Table 4. Adding the remaining
budget constraint on top does not improve accuracy further.
Mechanism.Invalid convolutional targets are not harmless noise: placing adapters on depthwise or unknown-
role layers actively destroys per-channel structure and drags accuracy down, so operator-validity filtering alone
recovers a meaningful fraction of the gap. The larger jump comes from semantic filtering, which keeps adapters
out of geometry-sensitive regression/DFL paths — this is the single largest source of stability improvement in
the ablation. Once both validity and safety are guaranteed, the remaining budget constraint mainly governs
compactness (adapter size), not accuracy, since the unsafe/invalid degrees of freedom have already been
removed from the search space.
Finding 5.The planner’s semantic constraints are not implementation details or defensive engineering
— they are the main source of the+0.041mAP 50:95 stability gain on YOLO12s (0.6900 → 0.7307), with
detection-head semantic filtering alone responsible for the largest share of that gain.
The detailed A1–A6 ablation tables remain in Appendix F; Table 7 keeps their main conclusions in the
experimental narrative. The suite tests whether the runtime contract, planner constraints, rank choice,
variant choice, training-side priors, and multi-adapter deployment are necessary design choices rather than
implementation details.
Rank sensitivity. Question:does higher rank simply mean higher accuracy, makingr=16an arbitrary
default?Evidence:on YOLO12s, mAP 50:95 moves from0.7288( r=8) to0 .7307( r=16) to0 .7363( r=32),
with trainable parameters and GFLOPs both increasing monotonically (Table 4).Mechanism:rank improves
accuracy monotonically but with diminishing returns — moving fromr=8to r=16yields a small gain
(+0.0019), while doubling again tor=32yields a similarly modest gain (+0.0056) at a substantially larger
parameter and GFLOP cost.Finding 6.Adapter rank controls an accuracy–cost trade-off rather than a free
lunch; r=16is a practical Pareto point balancing accuracy, adapter size, and inference cost, not a universally
optimal rank, and practitioners with a larger budget can trade further along this curve.
18

--- PAGE 19 ---

Tab. 8.Ablation A7 (key result): LM-inherited ranking vs. structure-aware plan. Numbers from W&B export, VOC
val2007, r=16, α=32,300ep. Gain is the∆between the structure-aware plan and the LM-inherited plan on the same
backbone.
Backbone LM-inherited plan Structure-aware plan Gain
Variant mAP 50:95 Variant mAP 50:95
YOLO11s DoRA (no rs)0.6479LoRA (rs, planner)0.7138+0.0659
YOLO12s DoRA (no rs)0.6112LoRA (rs, planner)0.7307+0.1195
RT-DETR-l DoRA (no rs)∼0.0(predicted)REFUSE(Full-SFT,0.6833) —∼+0.68
4.7 LM-inherited ranking vs. structure-aware placement
Question.This is the most direct test of the paper’s central claim: if a practitioner simply transplants
the PEFT variant ranking that performs best on language models directly onto a detector — ignoring
structure-aware placement — how much is lost?
Detailed ablations for contract management, planner constraints, rank sensitivity, local update variants,
RS-LoRA×DoRA interaction, and multi-task storage are reported in Appendix F.
Evidence.We transplant thebest-on-LMvariant ranking (DoRA ≻ LoRA ≻ LoHa) directly to detectors and
compare against Planner’s architecture-conditioned plan, using the same W&B-grounded numbers as Table 4.
On YOLO11s, the LM-inherited DoRA configuration gives only a marginal gain over Full-SFT (0.6479vs.
0.6428anchor), while the planner-selected RS-LoRA configuration improves substantially more (0.7138). On
YOLO12s, the same inherited choice becomes actively harmful,losing5.5mAP relative to Full-SFT (0.6112
vs.0 .6662), while the planner picks RS-LoRA and gains+0.0645. On RT-DETR-l, following the inherited
ranking would predict near-total collapse (Eq.(3)); the planner instead refuses and falls back to Full-SFT,
avoiding the catastrophe entirely.
Mechanism.The failure of the LM-inherited ranking is not caused by low-rank parameterization being
unsuitable for detectors in general — LoRA-family methods clearly work well when placed correctly (Table 4).
The failure is caused by carrying over an architecture-agnostic choice ofwhich variantandwhich targets
from the LM setting, where DoRA’s magnitude decomposition and RS-LoRA-off defaults were tuned against
Transformer linear-projection statistics that do not hold for YOLO’s heterogeneous operators.
Finding 7.The main failure mode of naive PEFT transfer from LLMs to detectors is not the low-rank
parameterization itself, but architecture-agnostic variant and target selection; structure-aware placement
recovers+0 .066to+0 .12mAP 50:95 over the LM-inherited default on stable backbones, and avoids catastrophic
collapse entirely on RT-DETR-l by refusing rather than following the inherited ranking.
5 Conclusion and Future Work
We presented YOLO-PEFT, a structure-aware PEFT framework that automates adapter placement across
heterogeneous detection graphs. The framework parses the detector graph into operator and semantic
roles (GraphParser), enumerates placements under operator-validity, head-semantics, and budget constraints
(Planner), and translates each placement into a YOLO-compatible execution contract that preserves checkpoint
compatibility, mergeability, and export compatibility (Contract). On the public v26.02 substrate [22], YOLO-
PEFT reproduces4 .7–8.1× on-disk compression,70–75%training-VRAM cut,87 .7%per-adapter distribution
savings, and95 .7%full-SFT mAP retention, while extending stable adaptation to attention-heavy and
multi-modal backbones that LM-style PEFT cannot reach. Ablations show that stable placement is strongly
architecture-conditioned: rankings inherited from LLMs are not just imperfect on detectors—they are actively
misleading. We invite the community to falsify the fingerprint→placement mapping by extending the45-cell
matrix to new architectures and tasks.
19

--- PAGE 20 ---

Appendices
A Merge Equivalence of the Fallback Manual Conv2d Backend
A.1 Dense convolution as im2col matrix multiplication
The fallback manual backend keeps the hostConv2d frozen and computes an additive LoRA branch on the
unfolded input patches. For a dense convolution,unfold maps x∈R N×C in×H×W to ˜x∈R N×C inkhkw×L,
where N is the batch size andL = H ′W ′. The unfold operator uses the same kernel size, stride, padding, and
dilation as the host convolution. Following the im2col convention, the convolution over unfolded patches is
computed as ˜x⊤W ⊤, producing anL×C out output before reshaping. A LoRA branch with rankr computes
(˜x⊤A)B⊤ and reshapes the result back to the convolutional output layout. If the host convolution carries a
bias term, the bias is preserved unchanged during merging.
A.2 Grouped convolution and per-group rank allocation
For a grouped convolution withGgroups, the fallback manual backend uses per-group factors
Ag ∈R (Cin/G)khkw×rg , B g ∈R Cout/G×rg ,
with total rank budgetr =P
g rg. The implementation uses the balanced allocationrg = r/G, so r must
be divisible by G. This avoids mixing channels across groups: because each group operates on a disjoint
input-channel block, concatenating per-group updates yields a block-diagonal update in the dense im2col
representation, thereby preserving the host convolution’s group structure. The dense case is recovered by
settingG= 1andr 1 =r.
A.3 Proof of Proposition 1
For each groupg, the LoRA branch is linear in the unfolded patches with effective matrixBgA⊤
g . Reshaping
this matrix to(Cout/G, Cin/G, k h, k w)yields
∆Wg =
 
BgA⊤
g

reshaped to(Cout/G, Cin/G, k h, k w).(4)
Concatenating the per-group updates gives a convolutional weight increment∆W of the same shape asW0.
Therefore
conv(W0, x) +s g
 
(Ag, Bg)g, x

=conv(W 0 +s∆W, x),
up to floating-point operation ordering. After applyingW0 ←W 0 + s∆W, the wrapper can be replaced by a
plainConv2d.
A.4 Numerical tolerance and implementation notes
The proof above covers the plain fallbackConv2d LoRA path. RS-LoRA and PEFT-managed variants are
delegated to the PEFT runtime; few-shot and adaptive fallback variants require the rank mask to be resolved
before applying the same merge pattern. The merged and unmerged paths differ only in operation ordering
(unfold + batched matrix multiplication versus a single convolution on the merged weight); outputs are
treated as equivalent within standard floating-point tolerance. Merge correctness is verified after wrapper
removal, not before. The equivalence holds in deployment/evaluation mode, where adapter dropout is disabled.
B Backend Routing and PEFT-Managed Variants
B.1 Backend selection rules
The PEFT backend is the default route for LoRA, RS-LoRA, DoRA, LoHa, LoKr, AdaLoRA, IA3, OFT,
BOFT, and HRA. The fallback manual backend is intentionally narrower: it is used for plainConv2d LoRA
wrapping when PEFT is unavailable, explicitly bypassed, or a fallback path is requested. Quantized paths are
delegated to the PEFT backend because quantization integration is backend-specific.
20

--- PAGE 21 ---

B.2 Quantized and wrapper-required paths
Wrapper-required training states are not considered export-compatible artifacts. If a wrapper cannot be
merged into a plain YOLO module or handled by a PEFT-managed export path, the contract layer refuses
export.
B.3 Export policy
Export is allowed only when the resulting model satisfies the runtime invariants of §3.5: unchanged base-
checkpoint loading, adapter-only checkpoints, ordinary YOLO module structure after merge, and export-
compatible modules for ONNX / TensorRT tooling. A PEFT-managed adapter is export-compatible only
after the PEFT merge path succeeds or after the contract verifies an equivalent unwrapped module structure.
C Runtime Metadata Schema
C.1 JSON fields
The current implementation persists runtime metadata rather than a full cryptographic manifest. The PEFT
path writesruntime_metadata.json alongside the PEFTsave_pretrained artifact; the fallback path writes
fallback_meta.json and a fallback weight file. Recorded fields include backend, variant, freeze-BN setting,
head-inclusion setting, target modules, and backend-specific runtime metadata.
C.2 Compatibility checks
At load time, YOLO-PEFT uses this metadata to select the PEFT or fallback loader and to verify that
the requested backend and target configuration are consistent with the saved adapter. Full base-checkpoint,
model-YAML, class-name, and architecture-hash checks are deploy-time extensions rather than current
hard-checking requirements.
C.3 Failure modes
The metadata check prevents backend/path confusion and missing fallback weights. Stronger graph, class-
vocabulary, and export-runtime mismatch checks require the full deploy-time manifest extension described in
§3.10.
D PEFT Matrix Details
The full core W&B measured matrix is reported as Table 4 in the main text. To avoid duplicating that table,
this appendix provides the numerical companion to Fig. 4 and the pairwise rank-agreement diagnostic.
D.1 Numerical companion to Fig. 4
Table 9 reports the compact∆mAP view corresponding to the heatmap. It separates the full-SFT anchor
from the PEFT deltas and marks catastrophic cells.
D.2 Pairwise rank agreement
Table 10 quantifies the rank-reversal pattern visible in Fig. 6. The within-CNN comparison is the only strongly
positive pair; crossing into attention-heavy, text-fusion, or Transformer-decoder detectors weakens or reverses
the variant ranking.
E Full Detector-Family Coverage Table
The main text reports aggregate detector-family coverage in Table 5. Table 11 enumerates the eleven backbones
underlying that aggregate without adding per-backbone measurements beyond the reported family status.
E.1 Full detector-family coverage
Closely related CNN-family detectors are listed separately here while the main text reports them as a single
architecture class. All CNN-family rows share the reported LoRA+decay acceptance band.
21

--- PAGE 22 ---

Tab. 9.Numerical companion to Fig. 4. Results are from the320-resolution diagnostic matrix. The Full-SFT row
reports the anchor mAP@0.5:0.95; all other rows report∆. Red entries are catastrophic (∆<−0.05). For collapsed
RT-DETR-l runs,∆ =−0.600denotes the clipped failure value used for visualization.
Variant YOLOv8n YOLO11n YOLO12n YOLO-W-s RT-DETR-l
Full-SFT (anchor)0.586 0.588 0.558 0.491 0.601
LoRA-vanilla+0.008 +0.011−0.086−0.097−0.229
LoRA+α-warmup+0.020 +0.017– – –
LoRA+ortho+0.019 +0.021−0.087−0.002−0.600
LoRA+decay+0.022 +0.022– – –
LoRA+ortho+decay+0.021 +0.020– – –
LoRA+spectral−0.009−0.007−0.084−0.033−0.600
DoRA+0.006 +0.007−0.081−0.292−0.600
LoHa+0.009 +0.012 +0.031 +0.081−0.600
LoKr+0.008−0.204−0.056−0.167−0.600
IA3 +0.007– – – –
HRA+0.002– – – –
BOFT−0.069– – – –
AdaLoRA0.000– – – –
Catastrophe rate1/14 0/10 6/7 4/7 7/7
Tab. 10.Pairwise Kendall’sτon∆mAP rankings. The within-CNN pair (YOLOv8n vs. YOLO11n) is the only pair
withp <0.05; all other pairs havep >0.3.
v8n 11n 12n W-s RT-DETR
v8n –+0.69 ∗ +0.33 +0.24 +0.00
11n –−0.14 +0.33 +0.18
12n –−0.05−0.18
W-s –−0.18
RT-DETR –
Tab. 11.Detector-family coverage underlying Table 5. CNN backbones share the reported LoRA+decay acceptance
band; non-CNN rows retain the explicit main-text status.
Backbone Architecture class Planner decision Reported status
YOLOv3 CNN LoRA+decay accept, within+0.018–+0.022band
YOLOv5n CNN LoRA+decay accept, within+0.018–+0.022band
YOLOv6n CNN LoRA+decay accept, within+0.018–+0.022band
YOLOv8n CNN LoRA+decay accept, within+0.018–+0.022band
YOLOv9c CNN LoRA+decay accept, within+0.018–+0.022band
YOLOv10n CNN LoRA+decay accept, within+0.018–+0.022band
YOLO11n CNN LoRA+decay accept, within+0.018–+0.022band
YOLO12n CNN+attention LoHa accept,+0.031
YOLO-World-s Text-fusion LoHa accept,+0.081
RT-DETR-l Transformer detectorREFUSEavoid−0.229
YOLO-Master MoE LoRA(exp) accept,+0.025
F Detailed Ablations
F.1 A1. Naive substitution vs. contract-managed substitution
Both backends in YOLO-PEFT use module substitution (the manual Conv2d backend for B2, PEFT-managed
wrappers for B1); the question is whether the surrounding contract management is required for deployment
compatibility. Table 12 compares a naive baseline with the full Contract layer. ONNX/TRT/Ckpt indicates
whether the trained model can be saved, reloaded, merged, and exported without manual graph surgery.
F.2 A2. Planner-component ablation
Table 13 ablates each constraint of the Planner planner on YOLO12s. The sequence isolates the contribution
of operator validity, detection-head semantic filtering, and budget pruning in turn.
22

--- PAGE 23 ---

Tab. 12.Ablation A1: naive substitution vs. contract-managed substitution (LoRAr=16, RS-LoRA, YOLO11s on
VOC).
Injection mode mAP 50:95 Trainable ONNX TRT Ckpt
Full-SFT (no LoRA)0.6428 9.44M✓ ✓ ✓
Naive substitution0.7138 10.44M× × ×
Contract layer (Contract)0.7138 10.44M✓ ✓ ✓
Tab. 13.Ablation A2: planner constraints on YOLO12s, VOC, LoRAr=16,α=32, RS-LoRA on,300ep.
Configuration Targets Trainable mAP 50:95 ∆
No constraint all-conv+linear+attn10.32M0.6900 +0.0238
+operator validity dense+linear+attn10.07M0.7094 +0.0432
+head semantics dense+linear+attn\reg10.06M0.7307 +0.0645
Full planner dense+attn (planner-pruned)10.06M0.7307+0.0645
Tab. 14.Ablation A3: rank sensitivity on YOLO12s, VOC,300ep. Anchor (Full-SFT):0.6662mAP 50:95 at9.26M
params.
rTrainable GFLOPs mAP 50:95 ∆
8 9.66M23.6 0.7288 +0.0626
16 10.06M25.6 0.7307 +0.0645
32 10.85M29.60.7363+0.0701
Tab. 15.Ablation A4: variant sweep under planner-selected placement,r=16,α=32, RS-LoRA on, VOC val2007.
Variant YOLO11s YOLO12s
mAP50:95 ∆mAP 50:95 ∆
LoRA0.7138 +0.0710 0.7307 +0.0645
DoRA0.7138 +0.0710– –
LoHa0.6788 +0.0359 0.7222 +0.0560
LoKr0.7033 +0.0605– –
IA3 0.6980 +0.0552 0.7210 +0.0548
HRA0.7276+0.0848 0.7453+0.0791
Tab. 16.Ablation A5: RS-LoRA×DoRA2×2factorial on YOLO12s, VOC, LoRAr=16,α=32,300ep.
DoRA
RS-LoRA off on
off 0.7094(+0.0432)0.7307(+0.0645)
on 0.6112(−0.0550)0.7138(+0.0476)
F.3 A3. Rank sensitivity
This ablation tests whether the planner’s default rank is a brittle choice. The sweep shows monotone gains
with rank while confirmingr=16as the parameter/GFLOP knee.
F.4 A4. Variant sweep under fixed placement
This ablation holds the planner placement fixed and varies only the local PEFT update rule. The ranking
shift across YOLO11s and YOLO12s supports the architecture-conditioned variant selection claim.
F.5 A5. RS-LoRA and DoRA interaction
This two-factor ablation checks whether training collapse is caused by the variant label alone or by interacting
training-side priors. Only the DoRA-without-RS-LoRA corner crosses the catastrophic threshold.
23

--- PAGE 24 ---

Tab. 17.Ablation A6: multi-task deployment economics. Storage (MB) forK=1, 5, 10, 20task adapters on YOLO11x
(114.6MB base,14.1MB adapter under the planner default).
MethodK=1K=5K=10K=20
Full-SFT per task114.6 573.0 1146.0 2292.0
Module-sub LoRA114.6 573.0 1146.0 2292.0
YOLO-PEFT (ours)128.7 185.1255.6 396.6
Savings vs. Full-SFT−12.3% 67.7%77.7%82.7%
F.6 A6. Multi-task adapter deployment
This ablation measures storage overhead when one base checkpoint serves multiple per-task adapters. The
K=10case is the main fleet-size example used in the compact deployment-economics summary of the main
text.
G FewShotLoRA Protocol
G.1 Motivation
Few-shot detection is the setting where PEFT regularisation should provide its strongest benefit; however, the
current artifact does not include completed VOC1/2/5/10-shot runs. This appendix specifies the evaluation
protocol for reproducibility; no empirical few-shot gains are claimed in the main experiments. Once the runs
are completed, this appendix can be extended with results and regulariser ablations.
G.2 Dataset split and evaluation
The planned evaluation uses VOC1-,2-,5-, and10-shot per-class splits, evaluates on VOC val2007, and runs
three random seeds. Primary reported fields are mAP50:95, mAP50, trainable parameters, peak memory, and
mean±std across seeds.
G.3 Backbones, baselines, and ablations
Planned backbones are YOLO11s, YOLO12s, and optionally YOLO-World-s as a text-fusion stress case.
Baselines include Full-SFT, head-only fine-tuning, standard LoRA, LoRA+dropout, and FewShotLoRA.
Component ablations remove DropConnect, distillation, adaptive rank, and variational rank independently.
H Reproducibility Notes
Experiments use VOC trainval07+12 for training and VOC val2007 for evaluation. Core W&B runs resize
images to640; the extended diagnostic matrix uses320; all runs use300epochs unless stated otherwise.
Optimisation uses AdamW with the learning rate, weight decay, momentum, cosine schedule, warmup, and
mosaic-close settings given in §4.1. Unless swept explicitly, adapters use rank16,α = 32, dropout0 .05,
RS-LoRA scaling enabled, and DoRA’s magnitude branch disabled. Runs use the A100 / CUDA / PyTorch /
PEFT stack reported in §4.1; exported W&B tables include backbone, variant, rank, alpha, scaling mode,
trainable parameters, mAP, memory, and adapter-size fields.
References
[1] Kerim Büyükakyüz. OLoRA: Orthonormal low-rank adaptation of large language models, 2024. arXiv preprint
arXiv:2406.01775.
[2] Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, and Ping Luo. AdaptFormer:
Adapting vision transformers for scalable visual recognition. InNeurIPS, 2022.
[3] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. YOLO-World: Real-time
open-vocabulary object detection. InCVPR, 2024.
[4] Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew
Zisserman. The PASCAL visual object classes challenge: A retrospective.International Journal of Computer
Vision, 111:98–136, 2015.
24

--- PAGE 25 ---

[5] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with
simple and efficient sparsity. InJMLR, volume 23, pages 1–39, 2022.
[6] Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, and Sai Qian Zhang. Parameter-efficient fine-tuning for large
models: A comprehensive survey, 2024.
[7] Soufiane Hayou, Nikhil Ghosh, and Bin Yu. LoRA+: Efficient low rank adaptation of large models. InICML,
2024.
[8] Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. Towards a unified view
of parameter-efficient transfer learning. InICLR, 2022.
[9] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo,
Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. InICML, 2019.
[10] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
Chen. LoRA: Low-rank adaptation of large language models. InICLR, 2022.
[11] Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanning Xu, Ee-Peng Lim, Roy Ka-Wei Lee, Lidong Bing, and Soujanya
Poria. LLM-Adapters: An adapter family for parameter-efficient fine-tuning of large language models. InarXiv
preprint 2304.01933, 2023.
[12] Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. LoraHub: Efficient cross-task
generalization via dynamic LoRA composition. InCOLM, 2024.
[13] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim.
Visual prompt tuning. InECCV, 2022.
[14] Shibo Jie and Zhi-Hong Deng. Convpass: Adapting vision transformers via convolutional bypasses. InIJCAI,
2023.
[15] Glenn Jocher, Ayush Chaurasia, Alex Stoken, et al. YOLOv5 by Ultralytics.https://github.com/ultralytics
/yolov5, 2020. GitHub repository, Accessed: 2026-05-10.
[16] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. YOLOv8 by Ultralytics.https://github.com/ultralytics/ult
ralytics, 2023. GitHub repository, Accessed: 2026-05-10.
[17] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. YOLO11 by Ultralytics.https://github.com/ultralytics/ult
ralytics, 2024. GitHub repository and documentation, Accessed: 2026-05-10.
[18] Damjan Kalajdzievski. A rank stabilization scaling factor for fine-tuning with LoRA, 2023. arXiv preprint
arXiv:2312.03732.
[19] Chuyi Li, Lulu Li, Hongliang Jiang, Kaiheng Weng, Yifei Geng, Liang Li, Zaidan Ke, Qingyuan Li, Meng Cheng,
Weiqiang Nie, et al. YOLOv6: A single-stage object detection framework for industrial applications, 2022. arXiv
preprint arXiv:2209.02976.
[20] Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized
focal loss: Learning qualified and distributed bounding boxes for dense object detection. InNeurIPS, 2020.
[21] Dongze Lian, Daquan Zhou, Jiashi Feng, and Xinchao Wang. Scaling & shifting your features: A new baseline for
efficient model tuning. InNeurIPS, 2022.
[22] Xu Lin, Jinlong Peng, Zhenye Gan, Jiawen Zhu, and Jun Liu. YOLO-Master: MOE-accelerated with specialized
transformers for enhanced real-time detection. https://github.com/Tencent/YOLO-Master , 2026. GitHub
repository; release v2026.02, Accessed: 2026-05-10.
[23] Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel.
Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. InNeurIPS, 2022.
[24] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and
Min-Hung Chen. DoRA: Weight-decomposed low-rank adaptation. InICML, 2024.
[25] Weiyang Liu, Zeju Qiu, Yao Feng, Yuliang Xiu, Yuxuan Xue, Longhui Yu, Haiwen Feng, Zhen Liu, Juyeon Heo,
Songyou Peng, Yandong Wen, Michael J. Black, Adrian Weller, and Bernhard Schölkopf. Parameter-efficient
orthogonal finetuning via butterfly factorization. InICLR, 2024.
[26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. InICLR, 2019.
25

--- PAGE 26 ---

[27] Wenyu Lv, Yi Zhao, Shangliang Xu, Jinman Wei, Guanzhong Wang, Cheng Cui, Yuning Du, Qingqing Dang, and
Yi Liu. RT-DETR: DETRs beat YOLOs on real-time object detection. InCVPR, 2024.
[28] Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, and Sayak Paul. PEFT: State-of-the-art
parameter-efficient fine-tuning methods.https://github.com/huggingface/peft, 2022. Accessed: 2026-05-10.
[29] Fanxu Meng, Zhaohui Wang, and Muhan Zhang. PiSSA: Principal singular values and singular vectors adaptation
of large language models. InNeurIPS, 2024.
[30] Hyeon-Woo Nam, Ye-Bin Moon, and Tae-Hyun Oh. FedPara: Low-rank hadamard product for communication-
efficient federated learning. InICLR, 2022.
[31] NVIDIA. TensorRT: High-performance deep learning inference optimizer and runtime.https://developer.nvid
ia.com/tensorrt, 2024. Accessed: 2026-05-10.
[32] ONNX Community. Open Neural Network Exchange (ONNX).https://onnx.ai/, 2021. Accessed: 2026-05-10.
[33] Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vulić, Sebastian Ruder, Kyunghyun Cho,
and Iryna Gurevych. AdapterHub: A framework for adapting transformers. InEMNLP Demos, 2020.
[34] Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. AdapterFusion:
Non-destructive task composition for transfer learning. InEACL, 2021.
[35] Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard
Schölkopf. Controlling text-to-image diffusion by orthogonal finetuning. InNeurIPS, 2023.
[36] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual
adapters. InNeurIPS, 2017.
[37] Joseph Redmon and Ali Farhadi. YOLOv3: An incremental improvement, 2018.
[38] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object
detection. InCVPR, 2016.
[39] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto,
Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. InNeurIPS, 2021.
[40] Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu,
Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, and Ion Stoica. S-LoRA: Serving thousands of concurrent
LoRA adapters. InMLSys, 2024.
[41] Yunjie Tian, Qixiang Ye, and David Doermann. YOLOv12: Attention-centric real-time object detectors. In
NeurIPS, 2025.
[42] Mojtaba Valipour, Mehdi Rezagholizadeh, Ivan Kobyzev, and Ali Ghodsi. DyLoRA: Parameter-efficient tuning of
pre-trained models using dynamic search-free low rank adaptation. InEACL, 2023.
[43] Simon Varailhon, Masih Aminbeidokhti, Marco Pedersoli, and Eric Granger. Source-free domain adaptation for
YOLO object detection. InComputer Vision – ECCV 2024 Workshops, pages 218–235. Springer, 2025. doi:
10.1007/978-3-031-91672-4_14.
[44] Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, and Guiguang Ding. YOLOv10: Real-time
end-to-end object detection. InNeurIPS, 2024.
[45] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. YOLOv9: Learning what you want to learn using
programmable gradient information. InECCV, 2024.
[46] Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui Tao, and Fu Lee Wang. Parameter-efficient fine-tuning
methods for pretrained language models: A critical review and assessment, 2023. arXiv preprint arXiv:2312.12148.
[47] Xudong Yao, Hao Liu, and Xiaoshan Yang. Component-coordinated and uncertainty-enhanced LoRA for few-shot
source-free domain adaptive object detection.Neurocomputing, 650:130787, 2025. doi: 10.1016/j.neucom.2025.13
0787.
[48] Shih-Ying Yeh, Yu-Guan Hsieh, Zhidong Gao, Bernard B. W. Yang, Giyeong Oh, and Yanmin Gong. Navigating
text-to-image customization: From LyCORIS fine-tuning to model evaluation. InICLR, 2024.
26

--- PAGE 27 ---

[49] Shen Yuan, Haotian Liu, and Hongteng Xu. Bridging the gap between low-rank and orthogonal adaptation via
householder reflection adaptation. InNeurIPS, 2024.
[50] Longteng Zhang, Lin Zhang, Shaohuai Shi, Xiaowen Chu, and Bo Li. LoRA-FA: Memory-efficient low-rank
adaptation for large language models fine-tuning, 2023. arXiv preprint arXiv:2308.03303.
[51] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao.
AdaLoRA: Adaptive budget allocation for parameter-efficient fine-tuning. InICLR, 2023.
[52] Han Zhou, Xingchen Wan, Ivan Vulić, and Anna Korhonen. AutoPEFT: Automatic configuration search for
parameter-efficient fine-tuning. InTACL, 2024.
27