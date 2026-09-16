# Omic：统一 R DE 与绘图的运行说明

## 范围与入口

Python 主入口是 `tools/omic_tools/omic_fetch_analysis_workflow.py`，函数
`omic_fetch_analysis_workflow(...)`。它负责取数、导出矩阵、调用 R DE、请求
Enrichr、调用 R 绘图及收集结果。`omic_analysis_components.py` 是内部编排组件。

- `tools/omic_tools/run_casestudy.R`：同一份既有 limma DE 与 case-study A–D 绘图。
- `enrichment/kegg_simple.R`：独立读取富集 CSV 并画图。
- `subprocess_r.py`：执行 R 子进程；`r_plotting.py`：检查 manifest 并收集产物。
- `casestudy_panels.Rmd` 留作迁移参考，正式入口不再依赖 knit/render。
- DE 仍以 metacell 为观测单位。没有切换为 donor-level DE。
- 性别专用 Rmd 的合并和历史性别图重画暂缓；不能称为已验收。
- 两张旧 Python 风格火山图已按用户决定取消。保留 case-study 新版火山图。

## Slurm 与运行环境

轻量文件操作可在 login 节点进行；Python/R、测试、绘图必须在已有 allocation
内执行。每次计算命令前重新查询，选择该用户当前运行中的合适作业，不创建新作业。
以下所有例子都从项目根目录执行。`OMIC_JOB_ID` 必须来自**紧接本次命令前**的查询，
不可把历史作业号写入脚本：

```bash
squeue --me -t RUNNING -o '%.18i %.9T %.30N %.60j'
read -r -p '当前可用 Job ID: ' OMIC_JOB_ID
```

若没有合适作业，停止计算。不要绕过 Slurm 在 login 节点运行。

本次验证使用 base Python 3.13.9、NumPy 2.5.3、SciPy 1.18.1、pytest 9.1.1；
R 4.5.2、limma 3.66.0、ggplot2 3.5.2、corrgram 1.15、ragg 1.4.0、jsonlite 2.0.0。
SciPy 从与 NumPy 不兼容的旧版本升级；ragg 在任务本地库重新编译，未修改共享 R 库。

当前工作区复现测试时需要以下环境设置；这些目录为本地依赖，**不随 Git 提交**。
其他机器需要准备自己的兼容安装，不能仅凭 Conda 环境名相同认定环境一致：

```bash
export OMIC_VALIDATION_ROOT="$PWD/webapp/sessions/alzheimer_test4/unified_pipeline_validation"
export R_LIBS="$OMIC_VALIDATION_ROOT/2026-09-15/task3-environment/r-library:/storage1/fs1/fuhai.li/Active/di.huang/cache/R"
export PYTHONPATH="$OMIC_VALIDATION_ROOT/2026-09-15/runtime/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
```

按前述方式查询、选择作业后，先确认计算节点环境：

```bash
srun --jobid="$OMIC_JOB_ID" --overlap --ntasks=1 --cpus-per-task=2 \
  Rscript -e 'cat(R.version.string,"\n"); for(p in c("limma","data.table","ggplot2","ragg","corrgram","vegan","patchwork","ggpubr","ggplotify","plotly","htmlwidgets","jsonlite")) {stopifnot(requireNamespace(p,quietly=TRUE));cat(p,as.character(packageVersion(p)),"\n")}'
```

`configs/paths.yaml` 为本地配置，不提交。保留现有外部数据配置，并核对这些键：

```yaml
analysis:
  r_script: tools/omic_tools/run_casestudy.R
enrichment:
  kegg_script: enrichment/kegg_simple.R
sessions:
  base: webapp/sessions
```

## 三种执行方式

### 1. 完整 Python workflow（会取数、计算 DE、请求在线富集）

这是后续新分析的用法，**不是本轮只重画验收时执行的命令**。
每次执行前重新查询并填写当前 `OMIC_JOB_ID`：

```bash
srun --jobid="$OMIC_JOB_ID" --overlap --ntasks=1 --cpus-per-task=2 \
  python tools/omic_tools/omic_fetch_analysis_workflow.py \
  --disease "Alzheimer's Disease" --cell-type astrocyte --organ brain \
  --session-id NEW_SESSION --sample-size 1000
```

查询标签必须与数据集定义一致。该例不保证重新取得历史 session 的相同样本。
`--sample-size` 是取样参数，**不是**富集的 top-N 参数。
`--no-plot` 关闭绘图但仍运行 DE/富集；`--no-de` 不是“读取已有结果重画”开关。
不要用入口的 `--run-tests` 代替下面的只重画验收，它会执行其他疾病查询。

函数调用可额外设置 `r_timeout=3600`（每次 R 调用）、`enrichment_timeout=60`
（每次 HTTP 请求）及 `enrichment_databases`。这三个参数没有对应的当前 CLI 选项。

### 2. 独立 R：全运行、仅 DE、仅重画

先设置绝对路径和与已有状态一致的分组标签：

```bash
export OMIC_SESSION=/absolute/path/to/existing_session
export OMIC_STATE="$OMIC_SESSION/casestudy_R/analysis_state.rds"
export OMIC_REDRAW=/absolute/path/to/new_redraw_directory
```

仅重画，不重新拟合 DE。每次执行前重新查询并选择作业：

```bash
srun --jobid="$OMIC_JOB_ID" --overlap --ntasks=1 --cpus-per-task=2 \
  Rscript tools/omic_tools/run_casestudy.R --stage plots \
  "$OMIC_SESSION" "$OMIC_REDRAW/casestudy_R" --state-file "$OMIC_STATE" \
  --ref-label normal --alt-label "Alzheimer's Disease" --input-scale linear_cp10k
```

默认从 session 寻找唯一的 `foranalysis_combined_normal_df_*.csv` 和
`foranalysis_combined_disease_df_*.csv`。有多组文件时必须显式传
`--ref-csv /absolute/ref.csv --alt-csv /absolute/alt.csv`。Python 桥接已显式传入路径。
重画校验输入文件摘要、标签和尺度；状态缺失或不匹配时报错，不自动重算 DE。

`--stage de` 只运行 DE/导出；`--stage all` 运行 DE 和 case-study 图。
不指定 stage 默认 `all`，因此**只重画必须显式写 `--stage plots`**。
独立 R 的 all 不包含在线 Enrichr 或独立 KEGG 绘图。

`COMPOSITE=false` 可关闭 ABCD 组合图；`WHICH_PC=PC1|PC2` 控制提琴图；
`PERMUTATIONS` 默认 999。`PSEUDOBULK` 默认关闭，本轮保持关闭；plots 阶段拒绝开启它。

### 3. 独立富集重画（读取 CSV，不请求 Enrichr）

`OMIC_ENRICHMENT` 指向已有 enrichment_results。直接使用原脚本的两个位置参数，
分别对 all/up/down 调用；每次执行前重新查询并选择作业。例如 up：

```bash
srun --jobid="$OMIC_JOB_ID" --overlap --ntasks=1 --cpus-per-task=2 \
  Rscript enrichment/kegg_simple.R \
  "$OMIC_ENRICHMENT/Alzheimer's_Disease_up_regulated" "$OMIC_REDRAW/plots/up"
```

down 输入改为对应 down_regulated 目录，输出改为 plots/down；all 输出为 plots。
Python 也是按此方式调度三次；保留原脚本 KEGG 点图 top20 与综合柱图每类 top5。
综合图包含 GO BP/CC/MF、KEGG、DisGeNET，不含 Reactome。已删除自行新增的
Reactome/KEGG 单库 top10 柱图代码及其额外 CLI 参数；所有富集 CSV 保留。

## 输入与 DE CSV 接口

矩阵：首列基因名，其他列为 metacell，至少两列观测。默认输入为线性 CP10K，
现有 R pipeline 使用 `log2(CP10K+1)` 和 limma moderated t-test。
已转换的输入须明确指定 `--input-scale log2_already`；不要重复取对数。

Python 使用 `de_results_io.read_de_results(path)` 读取以下五份 CSV：

```text
differential_expression/unpaired_differential_expression_results.csv
differential_expression/significant_genes_by_fdr.csv
differential_expression/significant_genes_by_fc.csv
differential_expression/significant_upregulated_genes.csv
differential_expression/significant_downregulated_genes.csv
```

列顺序和 dtype：

| 列 | pandas dtype | 含义 |
|---|---|---|
| Name | object | 基因名 |
| log2_fold_change | float64 | 原生 R logFC；alt−ref，在 log2 输入尺度上的差 |
| effect_size | float64 | 线性表达均值差除以 pooled SD + 1e-8；样本方差分母 n−1 |
| p_value | float64 | limma P.Value |
| FDR | float64 | 原 R 在受检验基因上校正的 adj.P.Val |
| is_significant | bool | FDR 小于 DE 阈值，默认 0.05 |
| abs_log2_fc | float64 | log2_fold_change 的绝对值 |

空子表保留表头；类型由读取适配器明确指定，裸 `pandas.read_csv` 不保证空表 dtype。
主表包含受检验基因，不把过滤掉的基因补成 p=1。
原生 R 结果和状态位于 `casestudy_R/DE_results_table.csv`、`analysis_state.rds`。

上层保留独立截断规则：all 按 FDR 取最多 1000；up/down 在显著基因中按
正/负 logFC 分组，各按 p_value 取最多 1000。all 不是 up/down 的拼接。
富集 CSV 默认每库保存前 50 条，完整响应在 raw 中；原绘图按前述 top20/top5 规则展示，
**不代表全部通过 FDR<0.05**。图中 Count 为该条目的命中基因数。

## 最终产物与状态

| 目录 | 产物 |
|---|---|
| casestudy_R | volcano、corrgram_genes、PCA_panel、figure_ABCD，各 PNG/PDF |
| casestudy_R | violin_PC1_2groups.pdf（或 PC2）；没有独立提琴图 PNG |
| casestudy_R | panel_A_labeled_genes.csv、panel_B_corrgram_genes.csv；没有独立 C/D 基因 CSV |
| plots | all 的 kegg_dotplot、pathway_combined_plot，各 PNG/HTML |
| plots/up、plots/down | 各组的同名 KEGG 点图、综合柱状图，各 PNG/HTML；依赖目录随 HTML 保留 |

case-study 文件名前缀为 `DE_results_`。有充分数据且 COMPOSITE=true 时，报告收集
10 张 PNG（case-study 4 + 富集 6）、6 个 HTML；`plot_paths` 为 HTML，`plots_for_report` 为 PNG 分类。
旧 `volcano_plots`、`enrichment_bar_plots` 分类保留为空，不再收集已取消的图。
case-study 的 PDF/CSV 保留在 manifest 中，不在 PNG 分类内。原富集脚本不输出 PDF。

R 非零退出、超时或缺产物都会报错。`de_success`、`enrichment_status`、
`de_plots_status`、`enrichment_plot_status` 分别表示各阶段，不只看 `kegg_success`
或文本 message。绘图失败保留已成功 DE 的状态，整体 success=False。
富集 empty 与 failed 区分；失败或空结果不会用旧图片冒充成功。HTML 为带相对依赖的文件，
单独拷贝 HTML 而遗漏依赖目录可能无法显示。

## 验收证据与边界

本地证据根目录为 `webapp/sessions/alzheimer_test4/unified_pipeline_validation/`；
这些数据、图片和测试日志不随代码提交，其他机器必须另行获取。

- Task 1/2 已合并 `7bac9cc`；Task 3 已合并 `425257a`。
- 原 R 与统一实现的真实输入：18,621 tested、13,467 significant；数值对照、五表 dtype、
  空结果、子进程失败证据见 `2026-09-15/review-evidence/`、`task2-review/`。
- 真实在线 Enrichr：`2026-09-15/online-enrichr-review/`，三组各 1000，14 库，
  3 POST + 42 GET 返回 HTTP200；不把 mock 的接口测试当作在线成功证据。
- 2026-09-16 绘图回归：`2026-09-16-panel-audit/regression.log`，19 passed，231.60 秒。
  真实 A/B 检查观察实际火山图标注层及 corrgram 标签；比较基因身份、顺序、输入表达矩阵、
  DE 数值和此前交付 CSV。证据在该目录 `regression/test_saved_ad_results_render_c0/`。
- 历史记录：曾修正自行新增的富集柱图方向，明确 `orientation="y"`，注释改为 FDR；
  `2026-09-16-enrichment-display/pytest.log` 为 11 passed，17.69 秒。
  用户随后要求仅复用原 R 绘图，该新增实现及其修复现已撤回，bars/ 不作为当前交付图。
- 切换为原 R 绘图之前的 Task 4 回归日志：`2026-09-16-panel-audit/task4-final.log`。
  最终输出 `19 passed in 230.84s`，0 failed、0 skipped，无 warning 汇总；
  已包含柱图方向修复和 A/B 内容核对。该套件不运行取数、在线请求或 DE 重算。
- 当前原 R 绘图三组调度验证：`2026-09-16-panel-audit/original-kegg-final.log`，
  `24 passed in 32.85s`，无失败、跳过或 warning 汇总。45 个富集输入文件内容及修改时间未变。
  原绘图逻辑复用于 all/up/down，分组输出放在同名产物目录的
  `test_existing_enrichment_uses_0/plots`、`plots/up`、`plots/down`。
  PNG 仅指定白色保存背景，避免透明背景在查看器中呈黑色，未改原图形配色/排序。

限制须保留：没有在最后版本重新执行“真实取数→DE→在线 Enrichr→全部图”的单次端到端运行。
各阶段已有实测，但不等于最后版本单次完整链路验收。没有逐份 PDF 的打开/渲染验收，
也没有所有 HTML 的浏览器交互验收；已检查相关文件非空、HTML 本地资源存在，并抽查 PNG。
独立提琴图 PNG、C/D 基因 CSV 不在原计划的具体产物清单内，当前未生成。
性别专用图及完整性别分析验收未执行。

### 只重画回归命令

设置 `OMIC_EXISTING_PLOT_SESSION` 为有匹配输入和 `casestudy_R/analysis_state.rds` 的
真实 AD session；设置 `OMIC_PANEL_REFERENCE_DIR` 为此前 A/B CSV 所在目录。
该真实数据测试使用 normal / Alzheimer's Disease 标签。缺少 session 环境变量会 skip，
**不能把 skip 当作真实验收成功**。每次执行前重新查询并选择作业：

```bash
srun --jobid="$OMIC_JOB_ID" --overlap --ntasks=1 --cpus-per-task=2 \
  python -m pytest tools/omic_tools/tests/test_r_plot_bridge.py \
  tools/omic_tools/tests/test_enrichment_r_plots.py \
  tools/omic_tools/tests/test_existing_plot_results.py -v -W error::UserWarning
```

不要直接改为运行整个 tests 目录来替代本命令：其他契约测试会拟合 DE，部分测试依赖外部服务。
