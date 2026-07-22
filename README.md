# MCTAP — Memory and Category-guided Tree of Attacks with Pruning

**MCTAP** は、LLM に対するジェイルブレイク攻撃を体系的に自動生成・評価する研究フレームワークである。
「攻撃生成」「防御層に基づく脆弱性分類」「マルチターン・エスカレーション」を統合した LLM 脆弱性評価基盤であり、カテゴリ別・モデル別の脆弱性プロファイルを定量的に明示することを目的とする。

> **論文タイトル**
> *Large Language Model に対する Jailbreak 攻撃の自動生成フレームワークと脆弱性分類の構築*

---

## 手法名について

| 略称        | 正式名称                                                    | 意味                                                          |
| --------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| **MCTAP** | Memory and Category-guided Tree of Attacks with Pruning | Memory（ゴール間知識転用）× Category（防御層に基づく脆弱性分類）× TAP（幅優先探索・枝刈り）の統合 |

MCTAP は既存の TAP (Mehrotra et al., 2023) を以下の3点で拡張する。

1. **Memory 層**：過去の成功攻撃をカテゴリ別ベクトル DB に蓄積し、ゴール間で知識を転用する
2. **Category 層**：LLM の処理層（トークン・命令・役割・意図・文脈）に基づく5カテゴリの脆弱性分類を導入し、探索を構造化する
3. **マルチターン・エスカレーション**：`PROGRESSIVE_MANIP` カテゴリ専用の、無害な文脈を先に構築してから段階的に有害な依頼へエスカレーションする多段階プロンプト構造と、会話履歴を維持したままエスカレーション文を最大 `multiturn_depth` 回まで言い換えて再挑戦するリトライループを備える

> **注**：3.のリトライループは、他カテゴリと同一の攻撃者プロンプト生成・スコアリングロジック（1〜4点評価）を、会話コンテキストを保持したまま繰り返し呼び出すものである。ターンごとに拒否理由を分析して構造化フィードバックを返す専用の診断機構は実装されていない。

---

## アーキテクチャ概要

```
【Memory 層】  ゴール間・長期探索
  過去成功例をカテゴリ別ベクトル DB に蓄積
  → 類似ゴールの成功プロンプトを hint として注入
        ↓
【TAP 層】     同一ゴール内・幅方向探索
  depth × root_width の探索木 + スコアによる枝刈り
  → カテゴリを適応的に選択（Exploit-Explore 制御）
        ↓
【PAIR 層】    ノード内・逐次改善
  history（過去 Attempt）を参照しながらプロンプトを反復改善

         ＋

【マルチターン・エスカレーション】  PROGRESSIVE_MANIP 専用
  ウォームアップ turn で無害な文脈を構築した後、
  エスカレーション turn を会話履歴を維持したまま
  最大 multiturn_depth 回リトライ（評価ロジックは他カテゴリと共通）
```

---

## 目次

- [実験結果](#実験結果)
- [脆弱性分類](#脆弱性分類)
- [ディレクトリ構成](#ディレクトリ構成)
- [環境構築](#環境構築)
- [モデル設定](#モデル設定)
- [実行方法](#実行方法)
- [出力フォーマット](#出力フォーマット)
- [評価指標](#評価指標)
- [設定パラメータ](#設定パラメータ)

---

## 実験結果

**ターゲットモデル**：Gemma-4-e2b、Llama-3.2-1B-Instruct、Qwen3.5-2B
**アタッカー**：Mistral-Nemo-Instruct-2407
**ベンチマーク**：Cyber Attack Facilitation カテゴリ（独自作製、各カテゴリ 40 ゴール × 5 = 200 ゴール）
**評価指標**：ASR / StrongREJECT / Binoculars（adversarial・original）/ 平均処理時間

### 全結果一覧

| モデル          | カテゴリ               | 件数  | Score4 | ASR       | SR    | Bino(adv) | Bino(orig) | dur(s) |
| ------------ | ------------------ | --- | ------ | --------- | ----- | --------- | ---------- | ------ |
| Gemma-4-e2b  | DIRECT\_OVERRIDE   | --- | ---    | ---       | ---   | ---       | ---        | ---    |
| Gemma-4-e2b  | ENC\_EVASION       | --- | ---    | ---       | ---   | ---       | ---        | ---    |
| Gemma-4-e2b  | INTENT\_CONCEAL    | 200 | 186    | 0.930     | 0.968 | 0.897     | 0.764      | 778    |
| Gemma-4-e2b  | PERSONA\_ROLEPLAY  | 200 | 190    | 0.950     | 0.968 | 0.886     | 0.763      | 718    |
| Gemma-4-e2b  | PROGRESSIVE\_MANIP | 200 | 199    | **0.995** | 0.975 | 0.847     | 0.765      | 552    |
| Llama-3.2-1B | DIRECT\_OVERRIDE   | 200 | 173    | 0.865     | 0.994 | 0.853     | 0.763      | 343    |
| Llama-3.2-1B | ENC\_EVASION       | 200 | 161    | 0.805     | 0.994 | 0.839     | 0.763      | 601    |
| Llama-3.2-1B | INTENT\_CONCEAL    | 200 | 131    | 0.655     | 0.985 | 0.897     | 0.763      | 452    |
| Llama-3.2-1B | PERSONA\_ROLEPLAY  | 200 | 179    | 0.895     | 0.955 | 0.847     | 0.765      | 298    |
| Llama-3.2-1B | PROGRESSIVE\_MANIP | 200 | 199    | **0.995** | 0.970 | 0.852     | 0.764      | 346    |
| Qwen3.5-2B   | DIRECT\_OVERRIDE   | 200 | 98     | 0.490     | 1.000 | 0.896     | 0.776      | 1562   |
| Qwen3.5-2B   | ENC\_EVASION       | 200 | 185    | 0.925     | 0.973 | 0.864     | 0.763      | 1536   |
| Qwen3.5-2B   | INTENT\_CONCEAL    | 200 | 196    | 0.980     | 0.954 | 0.920     | 0.766      | 114    |
| Qwen3.5-2B   | PERSONA\_ROLEPLAY  | 200 | 199    | 0.995     | 0.965 | 0.907     | 0.763      | 119    |
| Qwen3.5-2B   | PROGRESSIVE\_MANIP | 200 | 200    | **1.000** | 0.980 | 0.860     | 0.764      | 170    |

> SR: StrongREJECT、Bino(adv/orig): Binoculars adversarial/original prompt、dur: 平均処理時間（Score4・非Score4の加重平均）

### 主要な知見

- **`PROGRESSIVE_MANIP` が全モデルで最高 ASR**（Gemma: 0.995、Llama: 0.995、Qwen: 1.000）→ ウォームアップによる文脈構築とエスカレーション・リトループの組み合わせの有効性を示す（両者の寄与の切り分けは未検証、今後の課題）
- **Qwen3.5-2B の `DIRECT_OVERRIDE` への耐性**：ASR=0.490（98/200）と全実験中最低。他カテゴリ（0.925〜1.000）との乖離が顕著。SR=1.000 は Score4 到達応答が全て有害内容を含んでいたことを示す
- **Llama-3.2-1B の `INTENT_CONCEAL` への相対的耐性**：ASR=0.655（131/200）と他カテゴリより低く、意図分類器が相対的に機能
- **Binoculars の安定性**：Bino(adv) は 0.839〜0.920、Bino(orig) は 0.763〜0.776 の範囲。adversarial プロンプトが元プロンプトより一貫して高い自然言語らしさを示す
- **処理時間の差異**：Qwen の `DIRECT_OVERRIDE`（1562s）・`ENC_EVASION`（1536s）が突出して長い一方、`INTENT_CONCEAL`（114s）・`PERSONA_ROLEPLAY`（119s）は短時間で高 ASR を達成

---

## 脆弱性分類

MCTAP は LLM の処理パイプラインを5層に抽象化し、各層を標的とする攻撃を1カテゴリとして定義する（**defense-layer abstraction**）。

| カテゴリ                | 標的とする防御層            | ターン数     |
| ------------------- | ------------------- | -------- |
| `ENC_EVASION`       | トークナイザー・表層キーワードフィルタ | 1        |
| `DIRECT_OVERRIDE`   | システムプロンプト・優先順位制約    | 1        |
| `PERSONA_ROLEPLAY`  | 自己同一性・アイデンティティ制約    | 1        |
| `INTENT_CONCEAL`    | 意図分類器・文脈理解          | 1        |
| `PROGRESSIVE_MANIP` | 会話履歴の文脈依存安全評価       | マルチターン ★ |

★ `PROGRESSIVE_MANIP` のみマルチターン。無害な文脈を構築するウォームアップ turn の後、エスカレーション turn を会話履歴を維持したまま最大 `multiturn_depth` 回リトライする。評価（スコアリング）ロジック自体は他カテゴリと共通で、ターン単位の拒否理由分析・構造化フィードバック機構は持たない。

各カテゴリは **4 グループ × 5 バリアント = 20 パターン**（合計 100 パターン）に細分化。
グループが構造的多様性（攻撃の骨格）、バリアントが表面的多様性（語彙・フォーマット）を担う。

> 実装上、`VulnCategory` には `CONTEXT_INJECTION` も定義されているが、現在の実験・本ドキュメントでは使用しない（未使用カテゴリ）。

---

## ディレクトリ構成

```
MCTAP/
├── main.py                         # エントリポイント・設定
├── evaluate.py                     # 評価スクリプト (ASR / StrongREJECT / Binoculars)
├── requirements.txt
├── src/
│   ├── attack.py                   # MCTAP コア（PAIR × TAP × Memory × マルチターン）
│   ├── memory.py                   # RAG メモリストア (per-category .pt)
│   ├── models.py                   # データクラス・設定クラス
│   ├── vuln.py                     # 脆弱性カテゴリ定義（5カテゴリ × 100 パターン）
│   └── utils.py                    # ファイル I/O ユーティリティ
├── data/
│   ├── jailbreaks/                 # 成功した攻撃記録 (JSONL)
│   ├── traces/                     # 攻撃ツリーのトレース (JSON)
│   └── memory/                     # RAG ベクトルストア (.pt per category)
├── Dataset/
│   └── Cyber_Attack_Facilitation/  # ベンチマークデータセット (JSONL)
└── logs.txt                        # 実行ログ
```

---

## 環境構築

### 必要環境

| ツール       | バージョン   | 備考                     |
| --------- | ------- | ---------------------- |
| Python    | 3.10 以上 |                        |
| CUDA      | 11.8 以上 | GPU 使用時（Binoculars 計算） |
| LM Studio | 最新版     | ローカルモデル実行              |

> **重要：LM Studio の Context Length は 4096 に設定すること。**
> MCTAP はこの上限を前提に `max_tokens` を動的計算している。
> 変更した場合は `src/attack.py` の `_LM_STUDIO_CTX` も合わせて修正すること。

### インストール

```
# 1. リポジトリのクローン
git clone <repo_url>
cd MCTAP

# 2. 仮想環境の作成・有効化
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

# 3. 依存パッケージのインストール
pip install -r requirements.txt
```

### requirements.txt

```
# LLM / API
litellm>=1.30.0
openai>=1.0.0
anthropic>=0.25.0
huggingface_hub>=0.22.0
dspy-ai>=2.4.0

# データ処理
datasets>=2.18.0
torch>=2.1.0
jaxtyping>=0.2.28
pydantic>=2.0.0

# ユーティリティ
loguru>=0.7.0
tqdm>=4.66.0
transformers>=4.40.0   # Binoculars 計算
```

### 推奨モデル構成

| 役割        | 推奨モデル                                  | 備考       |
| --------- | -------------------------------------- | -------- |
| Attacker  | `mistralai/Mistral-Nemo-Instruct-2407` | 指示追従が安定  |
| Target    | `meta-llama/Llama-3.2-1B-Instruct`     | 軽量・高速    |
| Embedding | `nomic-ai/nomic-embed-text-v1.5`       | Memory 用 |
| Judge     | Attacker と同じモデルで可                      |          |

---

## モデル設定

`main.py` の `AttackConfig` でモデルを設定する。

### A. LM Studio（ローカル）

```
target_remote=Model(
    api_base="http://localhost:1234/v1",
    api_key="lm-studio",
    model_name="openai/llama-3.2-1b-instruct",
    temperature=0.0,
),
```

### B. Anthropic Claude API

```
target_claude=AnthropicModel(
    model_name="claude-sonnet-4-5",
    api_key="sk-ant-...",   # または環境変数 ANTHROPIC_API_KEY
    temperature=0.0,
),
```

### C. OpenAI API

```
target_remote=Model(
    api_base="https://api.openai.com/v1",
    api_key="sk-...",       # または環境変数 OPENAI_API_KEY
    model_name="openai/gpt-4o",
    temperature=0.0,
),
```

### D. HuggingFace Inference API

```
target_hf=HFModel(
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    api_key="hf_...",       # または環境変数 HF_TOKEN
    provider="hf-inference",
    temperature=0.0,
),
```

---

## 実行方法

### 攻撃実行

```
# main.py の AttackConfig を設定してから実行
python main.py
```

**Resume モード（途中から再開）**
`main.py` の `RESUME_MODE = True`（デフォルト）にすると、
既に処理済みのゴールをスキップして未処理分のみ実行する。
途中で落ちた場合もそのまま再実行すれば続きから始まる。

```
RESUME_MODE = True   # 再開モード（デフォルト）
RESUME_MODE = False  # 全件最初から実行
```

### 評価

```
python evaluate.py \
    --input  data/jailbreaks/C1/memory.jsonl \
    --output data/eval/C1_results.csv \
    --api-base http://localhost:1234/v1
```

---

## 出力フォーマット

### 成功記録 (JSONL)

`data/jailbreaks/{output_name}.jsonl` に 1 行 1 レコードで保存。
全カテゴリ共通で `attacker_prompt` は会話リスト形式。

```
{
  "attacker_prompt": [
    {"role": "user",      "content": "ウォームアップ T1"},
    {"role": "assistant", "content": "応答 T1"},
    {"role": "user",      "content": "エスカレーション（最終）"},
    {"role": "assistant", "content": "成功応答"}
  ],
  "conversation":        [ /* attacker_prompt と同じ内容 */ ],
  "target_model_answer": "成功応答（最終ターン）",
  "score":               4,
  "vuln_category":       "PROGRESSIVE_MANIP",
  "goal":                "攻撃ゴール",
  "duration":            42,
  "success":             true
}
```

### ターゲット応答 (JSONL)

`data/traces/{goal_slug}_responses.jsonl` に各ターンの応答を別ファイルで保存。
`Attempt` に応答テキストを持たせないことで `TreeNode` の deepcopy コストを削減。

```
{"child_id": "root",  "response": "初回応答"}
{"child_id": "1-0-2", "response": "ターゲット応答"}
```

### エラースキップ記録

```
data/jailbreaks/{output_name}_skipped.txt
```

例外によりスキップされたゴールを記録。Resume モード再実行時に自動的に対象となる。

---

## 評価指標

| 指標                           | 説明                                       |
| ---------------------------- | ---------------------------------------- |
| **ASR**（Attack Success Rate） | `stop_score ≥ 4` 到達率。LLM による有害性判定を使用     |
| **StrongREJECT**             | 拒否・具体性・説得力の3次元で 0〜1 をスコアリング。ASR の過大評価を補正 |
| **Binoculars（adversarial）**  | 攻撃プロンプトの自然言語らしさ。高いほど検出困難                 |
| **Binoculars（original）**     | 元の有害ゴールの自然言語らしさ（ベースライン）                  |

---

## 設定パラメータ

`main.py` の `AttackConfig` で設定。

| パラメータ               | デフォルト         | 説明                                 |
| ------------------- | ------------- | ---------------------------------- |
| `root_width`        | 5             | ルートノード数（並列スタート数）                   |
| `depth`             | 10            | 最大探索深度                             |
| `branching_factor`  | 3             | ノードあたりの子ノード数                       |
| `width`             | 10            | イテレーションごとに残すノード数                   |
| `stop_score`        | 4             | 成功と判定するスコア閾値（1〜4）                  |
| `multiturn_depth`   | 3             | `PROGRESSIVE_MANIP` の最大エスカレーション回数  |
| `exploit_threshold` | 3             | カテゴリ継続の最低スコア                       |
| `patience`          | 2             | カテゴリ切替までの低スコア許容回数                  |
| `memory_dir`        | `data/memory` | RAG ストアのディレクトリ（`None` で Memory 無効） |
| `vuln_categories`   | `None`        | 対象カテゴリのリスト。`None` の場合は `VulnCategory` 全メンバー（`CONTEXT_INJECTION` を含む）が対象になる点に注意。5カテゴリに限定する場合は明示的にリストで指定すること |

---

## 参考文献

- Mehrotra et al. (2023). *Tree of Attacks: Jailbreaking Black-Box LLMs Automatically.* NeurIPS 2024. arXiv:2312.02119
- Chao et al. (2023). *Jailbreaking Black Box LLMs in Twenty Queries.* arXiv:2310.08419
- Souly et al. (2024). *A StrongREJECT for Empty Jailbreaks.* NeurIPS 2024. arXiv:2402.10260
- Hans et al. (2024). *Binoculars: Accurate Zero-Shot Detection of LLM-Generated Text.* ICML 2024.

---

## 注意事項

本フレームワークは **LLM の安全性研究・レッドチーミング** を目的としている。
生成された攻撃プロンプトを実際の悪用に使用することは禁止する。
