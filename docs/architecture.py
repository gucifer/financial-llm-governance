"""
Financial LLM Governance Architecture Diagram
Sovereign AI orchestration for U.S. financial services — FS AI RMF aligned.

Regulatory anchors:
  - U.S. Treasury FS AI RMF (Feb 2026) — 230 control objectives
  - NIST AI RMF 1.0 (Jan 2023)
  - Executive Order 14110
  - OWASP LLM Application Security Top 10

Run:
    pip install diagrams
    python docs/architecture.py
Output: docs/financial_llm_governance_architecture.png
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import Lambda
from diagrams.aws.network import APIGateway
from diagrams.aws.security import Cognito, WAF
from diagrams.aws.database import ElasticacheForRedis
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import Sagemaker
from diagrams.aws.integration import SQS
from diagrams.aws.general import User
from diagrams.onprem.network import Kong
from diagrams.onprem.monitoring import Grafana, Datadog
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.mlops import Mlflow
from diagrams.azure.ml import AzureOpenAI
from diagrams.programming.language import Python

FONT = "Cascadia Code NF SemiBold"

graph_attrs = {
    "fontname": FONT,
    "fontsize": "13",
    "bgcolor":  "white",
    "pad":      "1.5",
    "splines":  "ortho",
    "nodesep":  "1.6",
    "ranksep":  "1.3",
}

node_attrs = {
    "fontname": FONT,
    "fontsize": "10",
    "width":    "1.6",
    "height":   "1.0",
}

edge_attrs = {
    "fontname":  FONT,
    "fontsize":  "9",
    "color":     "#444444",
}

# ── Legend HTML table ─────────────────────────────────────────────────────────
# Numbers: plain ASCII (circled Unicode not in Cascadia Code NF charset).
# Columns: 3 only (#, From -> To, What Happens) — Pillar already in cluster labels.
# No explicit TD WIDTH — let graphviz size to content to avoid overflow.
LEGEND = """<
<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="2" CELLPADDING="5" BGCOLOR="#f8f9fa">
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#1a1a2e"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="10"><B> # </B></FONT></TD>
    <TD ALIGN="CENTER" BGCOLOR="#1a1a2e"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="10"><B>From</B></FONT></TD>
    <TD ALIGN="CENTER" BGCOLOR="#1a1a2e"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="10"><B>To</B></FONT></TD>
    <TD ALIGN="CENTER" BGCOLOR="#1a1a2e"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="10"><B>What Happens</B></FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>1</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Clients</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Workload Router</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Tag by type (AML/KYC vs Reg Q&amp;A); assign data-sensitivity risk tier</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>2</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Workload Router</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Injection Detector</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Scan for adversarial inputs and OWASP LLM Top 10 #1 patterns; reject malicious requests</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>3</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Injection Detector</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Kong / JWT / PII Scrub</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Enforce JWT/OAuth 2.0; strip SSN, account numbers, ABA routing numbers at perimeter</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>4</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Kong layer</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Semantic Cache</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Check vector cache for semantically equivalent prior queries; cache hit bypasses LLM cost</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>5</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Semantic Cache</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">AWS API GW / WAF / Rate Limiter</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Cache miss: cross cloud boundary; WAF filters; rate limiter enforces quotas and writes audit log</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>6</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Rate Limiter</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">pgvector RAG Layer</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Reg Q&amp;A path: retrieve top-k chunks from FS AI RMF (230 controls), FINRA, SEC, FinCEN</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>7</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Rate Limiter</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Hybrid AML Model</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">AML/KYC path: route transaction narrative to supervised + unsupervised risk-scoring pipeline</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>8</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">pgvector</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Cross-Encoder Re-ranker</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Re-rank chunks by relevance; inject only top-k context into prompt to reduce hallucination</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>9</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Hybrid AML Model</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">FP Reduction Engine</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Filter false alerts; annotate high-risk transactions with SHAP scores (target: &lt;10% FPR)</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>10</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Re-ranker / FP Engine</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Azure OpenAI (GPT-4o)</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Both paths converge; grounded context (Reg Q&amp;A) or risk-annotated narrative (AML) sent for inference; no raw PII</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>11</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Azure OpenAI</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">ECLIPSE / PII Re-check / NeMo</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Parallel output validation: hallucination detection (arXiv:2512.03107), PII re-scan, STR schema check</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>12</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Output Validators</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Langfuse / SHAP-LIME / CW / SIEM</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Full LLM trace, FINRA audit JSON, CloudWatch metrics, SIEM security event written per response</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#2c3e50"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>13</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Observability layer</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">MLflow / Review Queue / Eval Trigger</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Write audit records to model registry; enqueue anomalous signals for human review; schedule re-eval runs</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="#c0392b"><FONT FACE="Cascadia Code NF SemiBold" COLOR="white" POINT-SIZE="12"><B>14</B></FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Eval Re-run Trigger</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">ECLIPSE Hallucination Detector (dashed)</FONT></TD>
    <TD><FONT FACE="Cascadia Code NF SemiBold" POINT-SIZE="9">Feedback loop: re-run hallucination suite on flagged responses; update model card in MLflow</FONT></TD>
  </TR>
</TABLE>>"""


with Diagram(
    "Financial LLM Governance — Sovereign AI Gateway Architecture",
    filename="financial_llm_governance_architecture",
    show=False,
    direction="TB",
    graph_attr=graph_attrs,
    node_attr=node_attrs,
    edge_attr=edge_attrs,
) as diag:

    # ── Row 0: Clients ────────────────────────────────────────────────────────
    with Cluster("Clients"):
        aml_client = User("AML / KYC\nWorkload")
        reg_client = User("Regulatory\nQ&A Workload")

    # ── Row 1: Input Classification ───────────────────────────────────────────
    with Cluster("Input Classification & Adversarial Defense\nOWASP LLM Top 10 #1  ·  FS AI RMF Pillar 2"):
        classifier = Python("Workload Router\n& Risk Scorer")
        adv_detect = Python("Prompt Injection\nDetector")

    # ── Row 2: Kong (on-prem DMZ) ─────────────────────────────────────────────
    with Cluster("Kong API Gateway  —  On-Prem / DMZ\nFS AI RMF Pillar 1 (Governance)  ·  Pillar 3 (Controls)\nSecrets Manager: API key vault · JWT signing"):
        kong      = Kong("Kong Gateway")
        jwt_auth  = Cognito("JWT / OAuth 2.0")
        pii_scrub = Python("PII Scrubbing\n(SSN · Acct · ABA)")

    # ── Row 3: Semantic Cache ─────────────────────────────────────────────────
    with Cluster("Semantic Cache  —  FS AI RMF Pillar 3"):
        sem_cache = ElasticacheForRedis("Redis + Embeddings\nSemantic Cache")

    # ── Row 4: AWS API Gateway ────────────────────────────────────────────────
    with Cluster("AWS API Gateway  —  Cloud Boundary\nFS AI RMF Pillar 2 (Risk ID)  ·  Pillar 4 (Monitoring)"):
        aws_apigw  = APIGateway("AWS API\nGateway")
        waf        = WAF("AWS WAF")
        rate_limit = Lambda("Rate Limiter\n& Request Logger")

    # ── Row 5: Parallel paths ─────────────────────────────────────────────────
    with Cluster("RAG Layer  —  Regulatory Q&A Path\nFS AI RMF Pillar 1 (Authoritative Grounding)"):
        pgvector = PostgreSQL("pgvector\nFS AI RMF · FINRA · SEC")
        reranker = Python("Cross-Encoder\nRe-ranker")

    with Cluster("AML / KYC Pipeline\nFS AI RMF Pillar 2 (Risk Scoring)"):
        aml_model  = Sagemaker("Hybrid AML Model\n(supervised + unsupervised)")
        fp_reducer = Python("False-Positive\nReduction  <10%")

    # ── Row 6: Azure OpenAI ───────────────────────────────────────────────────
    with Cluster("Azure OpenAI  —  Sovereign LLM Endpoint"):
        azure_oai = AzureOpenAI("Azure OpenAI\nGPT-4o")

    # ── Row 7: Output Validation ──────────────────────────────────────────────
    with Cluster("Output Validation & Guardrails\nFS AI RMF Pillar 3  ·  OWASP LLM Top 10 #2\nAWS KMS: encryption at rest for all outputs"):
        eclipse    = Python("ECLIPSE\nHallucination Detector")
        pii_check  = Python("PII Re-check\n(output scan)")
        guardrails = Python("NeMo Guardrails\n(STR schema enforcement)")

    # ── Row 8: Observability & XAI ────────────────────────────────────────────
    with Cluster("Observability, XAI & Audit\nFS AI RMF Pillar 4  ·  Pillar 1"):
        langfuse   = Grafana("Langfuse\nLLM Tracing")
        shap_lime  = Python("SHAP / LIME\n→ FINRA Audit JSON")
        cloudwatch = Cloudwatch("CloudWatch\nMetrics")
        siem       = Datadog("SIEM\nSplunk / Datadog")

    # ── Row 9: Governance & Feedback ─────────────────────────────────────────
    with Cluster("Model Governance & Feedback Loop\nFS AI RMF Pillar 1 (Accountability)"):
        mlflow    = Mlflow("MLflow Registry\n& Model Cards")
        feedback  = SQS("Human Review\nFeedback Queue")
        eval_loop = Python("Automated Eval\nRe-run Trigger")

    # ── Legend table node (raw graphviz HTML label) ───────────────────────────
    # rank=sink forces the legend subgraph below every other node.
    # Invisible edges from all three governance nodes ensure it stays anchored.
    with diag.dot.subgraph() as sink:
        sink.attr(rank="sink")
        sink.node("legend", label=LEGEND, shape="none", margin="0", fontname=FONT)

    diag.dot.edge(mlflow.nodeid,    "legend", style="invis")
    diag.dot.edge(feedback.nodeid,  "legend", style="invis")
    diag.dot.edge(eval_loop.nodeid, "legend", style="invis")

    # ── Edges (numbered ① – ⑭) ───────────────────────────────────────────────
    # Rule: only ONE edge per numbered step carries the label.
    # Unlabeled sibling edges keep nodes on the same graphviz rank
    # without duplicating the number.
    N = {"fontsize": "17", "fontname": FONT, "fontcolor": "#1a1a2e"}

    # 1 — one labeled edge from clients; second client unlabeled (same rank)
    aml_client >> Edge(label="1", **N) >> classifier
    reg_client >> classifier

    # 2
    classifier >> Edge(label="2", **N) >> adv_detect

    # 3 — label only to kong; unlabeled to jwt_auth and pii_scrub
    adv_detect >> Edge(label="3", **N) >> kong
    adv_detect >> jwt_auth
    adv_detect >> pii_scrub

    # 4 — label only from kong; unlabeled from the others
    kong      >> Edge(label="4", **N) >> sem_cache
    jwt_auth  >> sem_cache
    pii_scrub >> sem_cache

    # 5 — label only to aws_apigw; unlabeled to waf and rate_limit
    sem_cache >> Edge(label="5  cache miss", **N) >> aws_apigw
    sem_cache >> waf
    sem_cache >> rate_limit

    # 6 / 7
    rate_limit >> Edge(label="6  Reg Q&A",   **N) >> pgvector
    rate_limit >> Edge(label="7  AML / KYC", **N) >> aml_model

    # 8 / 9
    pgvector  >> Edge(label="8", **N) >> reranker
    aml_model >> Edge(label="9", **N) >> fp_reducer

    # 10 — label only from reranker; unlabeled from fp_reducer
    reranker   >> Edge(label="10", **N) >> azure_oai
    fp_reducer >> azure_oai

    # 11 — label only to eclipse; unlabeled to pii_check and guardrails
    azure_oai >> Edge(label="11", **N) >> eclipse
    azure_oai >> pii_check
    azure_oai >> guardrails

    # 12 — label only from eclipse; unlabeled from pii_check and guardrails
    eclipse    >> Edge(label="12", **N) >> langfuse
    pii_check  >> langfuse
    guardrails >> langfuse
    guardrails >> shap_lime
    guardrails >> cloudwatch
    guardrails >> siem

    # 13 — label only from langfuse; unlabeled from the rest
    langfuse   >> Edge(label="13", **N) >> mlflow
    shap_lime  >> mlflow
    cloudwatch >> mlflow
    siem       >> mlflow
    siem       >> feedback
    siem       >> eval_loop

    # 14 — feedback loop (dashed, red) — constraint=False prevents this back-edge
    # from pulling eclipse down to the rank of eval_loop in the layout engine
    eval_loop >> Edge(
        label="14  re-eval", style="dashed",
        constraint="False",
        fontsize="17", fontname=FONT, fontcolor="#c0392b",
        color="#c0392b",
    ) >> eclipse
