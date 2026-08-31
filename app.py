import streamlit as st
from datetime import datetime

# ============================================================
# AGENT RUNTIME INTEGRATION
# ============================================================
# The real agent runtime (router -> agent -> runtime -> Ollama -> tools)
# is wired into the existing UI below. Loading is best-effort so the
# rest of the frontend keeps working even if the runtime is unavailable.
try:
    from demo import run_task, DemoError

    _RUNTIME_AVAILABLE = True
except Exception:  # pragma: no cover - runtime import failure
    _RUNTIME_AVAILABLE = False


def render_execution_trace(events):
    """Render the serialized event trace as an expandable list."""
    for event in events:
        etype = event.get("type", "")
        detail = " · ".join(
            f"{k}={v}"
            for k, v in event.items()
            if k not in ("type", "execution_id") and v not in (None, "")
        )
        st.markdown(
            f'<div class="card" style="padding:8px 12px;margin-bottom:5px">'
            f'<span class="purple">▸</span> <b>{etype}</b>'
            f'<span class="muted" style="float:right">{detail}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sovereign AI Workbench",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# OPENCODE-INSPIRED THEME
# ============================================================

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: #0d0b0b;
        color: #d7d2cd;
    }

    [data-testid="stHeader"] {
        background: #0d0b0b;
        border-bottom: 1px solid #292323;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #100d0d;
        border-right: 1px solid #302a2a;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.7rem;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer {
        visibility: hidden;
    }

    /* General typography */
    html, body, [class*="css"] {
        font-family: "Courier New", monospace;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* OpenCode-like logo */
    .brand {
        font-family: "Courier New", monospace;
        font-size: 27px;
        font-weight: 900;
        letter-spacing: -2px;
        color: #f0ece8;
        margin-bottom: 2px;
    }

    .brand-sub {
        color: #706a67;
        font-size: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 18px;
    }

    /* Sidebar navigation */
    .nav-label {
        color: #817a76;
        font-size: 10px;
        letter-spacing: 1.5px;
        margin: 17px 0 6px 4px;
        text-transform: uppercase;
    }

    /* Buttons */
    .stButton > button {
        background: #171313;
        color: #d8d2ce;
        border: 1px solid #342e2e;
        border-radius: 4px;
        min-height: 38px;
        font-family: "Courier New", monospace;
        font-size: 12px;
        text-align: left;
    }

    .stButton > button:hover {
        border-color: #7566d9;
        color: #ffffff;
        background: #1c1818;
    }

    /* Inputs */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div,
    .stFileUploader section {
        background: #141111 !important;
        color: #ddd7d3 !important;
        border-color: #302a2a !important;
        border-radius: 4px !important;
        font-family: "Courier New", monospace !important;
    }

    /* Cards */
    .card {
        background: #141111;
        border: 1px solid #2b2525;
        border-radius: 5px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .card:hover {
        border-color: #413939;
    }

    .eyebrow {
        color: #817975;
        font-size: 10px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .page-title {
        color: #f0ece8;
        font-size: 30px;
        line-height: 1.15;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .page-title span {
        color: #b6a6ff;
    }

    .page-desc {
        color: #8f8884;
        font-size: 12px;
        line-height: 1.7;
        max-width: 760px;
        margin-bottom: 26px;
    }

    .section-title {
        color: #c9c2bd;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        margin: 20px 0 9px;
    }

    .metric {
        background: #141111;
        border: 1px solid #2b2525;
        border-radius: 5px;
        padding: 15px;
        min-height: 100px;
    }

    .metric-name {
        color: #746d69;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #eee9e5;
        font-size: 25px;
        font-weight: 700;
        margin: 8px 0 3px;
    }

    .muted {
        color: #6f6864;
        font-size: 10px;
    }

    .good {
        color: #79c9a1;
    }

    .warn {
        color: #d8ad6d;
    }

    .bad {
        color: #df7f88;
    }

    .purple {
        color: #b6a6ff;
    }

    .terminal {
        background: #090808;
        border: 1px solid #2a2525;
        border-radius: 5px;
        padding: 15px;
        color: #a8a19d;
        font-family: "Courier New", monospace;
        font-size: 11px;
        line-height: 1.7;
        min-height: 250px;
    }

    .terminal-head {
        color: #eee9e5;
        border-bottom: 1px solid #282222;
        padding-bottom: 9px;
        margin-bottom: 10px;
    }

    .status-line {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #211d1d;
        padding: 10px 0;
        font-size: 11px;
    }

    .pipeline {
        text-align: center;
        background: #141111;
        border: 1px solid #2b2525;
        border-radius: 5px;
        padding: 13px 8px;
    }

    .pipeline-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 27px;
        height: 27px;
        border-radius: 50%;
        background: #211c35;
        color: #b6a6ff;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .pipeline-name {
        color: #d6d0cc;
        font-size: 10px;
        font-weight: 700;
    }

    .footer-line {
        border-top: 1px solid #282222;
        margin-top: 40px;
        padding-top: 13px;
        color: #625b58;
        font-size: 9px;
    }

    /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio label {
        color: #aaa39e !important;
        font-size: 11px !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        color: #ffffff !important;
    }

    /* Mobile */
    @media (max-width: 800px) {
        .page-title {
            font-size: 24px;
        }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

def header(eyebrow, title, accent, description):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-title">{title} <span>{accent}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="page-desc">{description}</div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def metric_card(name, value, detail="", status_class=""):
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-name">{name}</div>
            <div class="metric-value {status_class}">{value}</div>
            <div class="muted">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_row(name, value, cls=""):
    st.markdown(
        f"""
        <div class="status-line">
            <span>{name}</span>
            <span class="{cls}">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR — OPENCODE-INSPIRED
# ============================================================

with st.sidebar:
    st.markdown('<div class="brand">opencode</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">sovereign ai / sih 26117</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)

    page = st.radio(
        "Workspace",
        [
            "Dashboard",
            "AI Workbench",
            "Document AI",
            "Knowledge Base",
            "Agent Orchestrator",
            "Code Lab",
            "Multimodal AI",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="nav-label">System</div>', unsafe_allow_html=True)

    system_page = st.radio(
        "System",
        ["Network Monitor", "System Monitor", "Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("LOCAL RUNTIME")
    st.markdown("🟢 **SOVEREIGN MODE**")
    st.caption("External API calls: 0")

# If system navigation is selected, it wins.
if system_page != "Network Monitor" or page == "Dashboard":
    if system_page != "Network Monitor" and page != "Dashboard":
        selected_page = system_page
    elif page == "Dashboard":
        selected_page = "Dashboard"
    else:
        selected_page = page
else:
    selected_page = page

# ============================================================
# DASHBOARD
# ============================================================

if selected_page == "Dashboard":
    header(
        "PRIVATE • LOCAL • MULTIMODAL • AGENTIC",
        "Sovereign AI",
        "Workbench",
        "A local-first AI workspace for confidential documents, intelligent agents, multimodal reasoning and sandboxed development.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("External Calls", "0", "All network access blocked", "good")
    with c2:
        metric_card("Local Models", "3", "Ready for inference", "purple")
    with c3:
        metric_card("Documents", "247", "Indexed locally")
    with c4:
        metric_card("Knowledge Chunks", "18,492", "Vector index ready")

    section("QUICK START")

    q1, q2, q3 = st.columns(3)

    with q1:
        st.markdown(
            """
            <div class="card">
                <div class="eyebrow">01 / AI</div>
                <b>AI Workbench</b>
                <p class="muted">Run prompts against a local model without sending data outside the system.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q2:
        st.markdown(
            """
            <div class="card">
                <div class="eyebrow">02 / DOCS</div>
                <b>Document AI</b>
                <p class="muted">Upload inspection reports and prepare them for OCR, extraction and reasoning.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q3:
        st.markdown(
            """
            <div class="card">
                <div class="eyebrow">03 / AGENTS</div>
                <b>Agent Orchestrator</b>
                <p class="muted">Inspect a transparent multi-step workflow from document to approval note.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section("RUNTIME")

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown(
            """
            <div class="terminal">
                <div class="terminal-head">◉ local-runtime / activity</div>
                <div><span class="good">19:01:22 LOCAL</span> &nbsp; Document loaded</div>
                <div><span class="good">19:01:24 LOCAL</span> &nbsp; OCR completed</div>
                <div><span class="good">19:01:26 LOCAL</span> &nbsp; Knowledge search</div>
                <div><span class="good">19:01:30 LOCAL</span> &nbsp; Model inference</div>
                <div><span class="good">19:01:34 LOCAL</span> &nbsp; Document generated</div>
                <div><span class="bad">19:01:40 BLOCKED</span> &nbsp; External endpoint request</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        status_row("Sovereign Mode", "ENABLED", "good")
        status_row("Network Isolation", "ACTIVE", "good")
        status_row("Sandbox", "ACTIVE", "good")
        status_row("Local Storage", "READY", "good")
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# AI WORKBENCH
# ============================================================

elif selected_page == "AI Workbench":
    header(
        "PRIVATE AI WORKSPACE",
        "AI",
        "Workbench",
        "Interact with local AI models and process confidential information without external API calls.",
    )

    left, right = st.columns([1.45, 0.75])

    with left:
        section("PROMPT CONSOLE")

        prompt = st.text_area(
            "Your prompt",
            placeholder="Ask your local AI model something...",
            height=180,
            label_visibility="collapsed",
        )

        uploaded = st.file_uploader(
            "Attach confidential document",
            type=["pdf", "docx", "txt", "csv"],
        )

        if st.button("▶  Run Local AI", use_container_width=True):
            if prompt.strip():
                st.success("Demo inference completed locally.")
                st.markdown(
                    """
                    <div class="card">
                        <div class="eyebrow">LOCAL AI RESPONSE</div>
                        <p>This is a frontend demonstration. Connect Ollama, llama.cpp or another local inference server here.</p>
                        <div class="good">● DATA STAYS LOCAL</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Please enter a prompt.")

    with right:
        section("MODEL ROUTER")

        model = st.selectbox(
            "Local model",
            ["Automatic Selection", "Qwen2.5-7B", "Llama 3.2", "Mistral 7B"],
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">ACTIVE MODEL</div>', unsafe_allow_html=True)
        st.markdown(f"### {model}")
        status_row("Inference", "LOCAL", "good")
        status_row("External calls", "0", "good")
        status_row("Data leaving system", "0 MB", "good")
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# DOCUMENT AI
# ============================================================

elif selected_page == "Document AI":
    header(
        "DOCUMENT INTELLIGENCE",
        "Document",
        "AI",
        "Upload inspection reports and confidential documents for local OCR, extraction, analysis and generation.",
    )

    left, right = st.columns(2)

    with left:
        section("DOCUMENT UPLOAD")
        file = st.file_uploader(
            "Upload your document",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
        )

        if file:
            st.success(f"Uploaded: {file.name}")
            st.caption(f"File size: {round(file.size / 1024, 2)} KB")

            if st.button("⌕  Analyze Document", use_container_width=True):
                st.success("Demo document analysis completed.")

    with right:
        section("EXTRACTED INFORMATION")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        status_row("Inspection Date", "28 Aug 2026")
        status_row("Location", "Plant A")
        status_row("Issues Found", "7")
        status_row("Critical Issues", "2", "bad")
        status_row("Risk Level", "HIGH", "bad")
        st.markdown("</div>", unsafe_allow_html=True)

    section("AI DOCUMENT PIPELINE")

    pipeline = [
        ("1", "OCR", "COMPLETED"),
        ("2", "KNOWLEDGE", "COMPLETED"),
        ("3", "REASONING", "READY"),
        ("4", "GENERATE", "READY"),
        ("5", "VALIDATE", "READY"),
    ]

    cols = st.columns(5)
    for col, (num, name, status) in zip(cols, pipeline):
        with col:
            st.markdown(
                f"""
                <div class="pipeline">
                    <div class="pipeline-num">{num}</div>
                    <div class="pipeline-name">{name}</div>
                    <div class="good" style="font-size:9px;margin-top:5px">{status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("▣  Generate Approval Note", use_container_width=True):
        st.success("Demo Approval_Note.docx generated.")

# ============================================================
# KNOWLEDGE BASE
# ============================================================

elif selected_page == "Knowledge Base":
    header(
        "ORGANIZATIONAL MEMORY",
        "Knowledge",
        "Base",
        "Search internal SOPs, manuals, reports and organizational knowledge using a private local retrieval system.",
    )

    query = st.text_input(
        "Search Knowledge Base",
        placeholder="Search SOPs, manuals, reports...",
    )

    if st.button("⌕  Search Knowledge", use_container_width=True):
        if query.strip():
            st.success(f"Searching local knowledge for: {query}")
            st.markdown(
                """
                <div class="card">
                    <div class="eyebrow">SEARCH RESULT</div>
                    <b>Inspection SOP — Section 4.2</b>
                    <p class="muted">Relevant procedure found in the local knowledge base.</p>
                    <span class="good">Relevance Score: 94%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section("INDEXED DOCUMENTS")

    docs = [
        ("Safety_Manual.pdf", "2.4 MB"),
        ("Inspection_SOP.pdf", "1.8 MB"),
        ("Previous_Approval_Notes.docx", "840 KB"),
        ("Operational_Guidelines.pdf", "3.2 MB"),
        ("Plant_A_Checklist.xlsx", "1.1 MB"),
    ]

    for name, size in docs:
        st.markdown(
            f"""
            <div class="card" style="padding:12px 15px">
                <b>▱ {name}</b>
                <span style="float:right" class="muted">{size} · <span class="good">Indexed ✓</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("247 documents indexed · 18,492 knowledge chunks")

# ============================================================
# AGENT ORCHESTRATOR
# ============================================================

elif selected_page == "Agent Orchestrator":
    header(
        "AUTONOMOUS WORKFLOW",
        "Agent",
        "Orchestrator",
        "Create transparent AI workflows where every step, tool and decision can be inspected.",
    )

    left, right = st.columns([1.35, 0.75])

    steps = [
        "Read document",
        "OCR extraction",
        "Search local knowledge base",
        "Identify relevant SOP",
        "Reason over findings",
        "Draft approval note",
        "Validate output",
        "Generate DOCX",
    ]

    with left:
        section("INSPECTION REPORT AGENT")

        for i, step in enumerate(steps, 1):
            st.markdown(
                f"""
                <div class="card" style="padding:11px 13px;margin-bottom:7px">
                    <span class="purple">{i:02d}</span>
                    <span style="margin-left:12px">{step}</span>
                    <span class="good" style="float:right">✓</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("▶  Run Agent Workflow", use_container_width=True):
            st.success("Agent workflow started successfully.")

    with right:
        section("TOOLS")

        tools = [
            "Local OCR",
            "Vector Search",
            "Document Generator",
            "Sandbox",
            "Network Monitor",
        ]

        for tool in tools:
            st.markdown(
                f"""
                <div class="card" style="padding:11px 13px">
                    🛡 {tool}
                    <span class="good" style="float:right">ALLOWED</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section("AGENT RUNTIME")

    if not _RUNTIME_AVAILABLE:
        st.warning(
            "Agent runtime is not available (demo package failed to load). "
            "The rest of the workbench remains fully functional."
        )
    else:
        task_input = st.text_area(
            "Agent task",
            value="Use the calculator to calculate 25 + 17.",
            height=90,
            label_visibility="collapsed",
            key="runtime_task",
        )

        run_clicked = st.button(
            "▶  Run Agent Runtime",
            use_container_width=True,
            key="runtime_run",
        )

        if run_clicked:
            if not task_input.strip():
                st.warning("Please enter a task.")
            else:
                with st.spinner("Routing agent · executing …"):
                    try:
                        result = run_task(task_input.strip())
                    except DemoError as error:
                        st.error(str(error))
                        result = None

                if result:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    exec_cols = st.columns([1, 1, 1, 1])
                    exec_cols[0].markdown(
                        f'<div class="eyebrow">SELECTED AGENT</div><b>{result["agent"]}</b>',
                        unsafe_allow_html=True,
                    )
                    exec_cols[1].markdown(
                        f'<div class="eyebrow">EXECUTION ID</div>'
                        f'<span class="muted">{result["execution_id"] or "—"}</span>',
                        unsafe_allow_html=True,
                    )
                    exec_cols[2].markdown(
                        f'<div class="eyebrow">STATUS</div>'
                        f'<span class="good">{result["status"]}</span>',
                        unsafe_allow_html=True,
                    )
                    exec_cols[3].markdown(
                        f'<div class="eyebrow">STEPS</div><b>{result["steps"]}</b>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown('<div class="eyebrow">FINAL RESULT</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="card"><div class="page-title" style="font-size:20px">'
                        f'{result["result"]}</div></div>',
                        unsafe_allow_html=True,
                    )

                    with st.expander("Execution trace"):
                        render_execution_trace(result["events"])

# ============================================================
# CODE LAB
# ============================================================

elif selected_page == "Code Lab":
    header(
        "LOCAL DEVELOPMENT",
        "Code",
        "Lab",
        "Use a local AI coding assistant with a sandboxed execution environment.",
    )

    code = st.text_area(
        "Code Editor",
        value="""# Example Python program

numbers = [4, 8, 15, 16, 23, 42]

for number in numbers:
    print(number)
""",
        height=300,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        run = st.button("▶  Run Code", use_container_width=True)
    with c2:
        explain = st.button("✦  Explain Code", use_container_width=True)
    with c3:
        tests = st.button("🧪  Generate Tests", use_container_width=True)

    if run:
        st.markdown(
            """
            <div class="terminal">
                <div class="terminal-head">◉ sandbox / terminal</div>
                <div>> Running sandbox...</div>
                <div>4</div>
                <div>8</div>
                <div>15</div>
                <div>16</div>
                <div>23</div>
                <div>42</div>
                <br>
                <div class="good">✓ Execution completed</div>
                <div class="good">✓ Tests passed: 8/8</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if explain:
        st.info("Demo: the code iterates through a list and prints each number.")

    if tests:
        st.success("Demo: generated test plan for the example program.")

# ============================================================
# MULTIMODAL AI
# ============================================================

elif selected_page == "Multimodal AI":
    header(
        "VISION + LANGUAGE",
        "Multimodal",
        "AI",
        "Analyze images, diagrams, charts and visual documents using local vision-language models.",
    )

    left, right = st.columns(2)

    with left:
        image = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "webp"],
        )

        if image:
            st.image(image, caption="Uploaded visual", use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="card">
                <div class="eyebrow">AI VISUAL ANALYSIS</div>
                <h3>Ready for visual reasoning</h3>
                <p class="muted">Upload an image to extract objects, text, relationships and actionable insights.</p>
                <div class="good">● Local Vision Model Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# NETWORK MONITOR
# ============================================================

elif selected_page == "Network Monitor":
    header(
        "DATA SOVEREIGNTY",
        "Network",
        "Monitor",
        "Monitor network activity and demonstrate that confidential information stays within the system.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("External Calls", "0", "Blocked", "good")
    with c2:
        metric_card("Data Sent", "0 MB", "Outside system", "good")
    with c3:
        metric_card("Data Received", "0 MB", "Outside system", "good")
    with c4:
        metric_card("Local Requests", "128", "Internal runtime")

    section("ACTIVITY LOG")

    logs = [
        ("19:01:22", "LOCAL", "Document loaded"),
        ("19:01:24", "LOCAL", "OCR completed"),
        ("19:01:26", "LOCAL", "Knowledge search"),
        ("19:01:30", "LOCAL", "Model inference"),
        ("19:01:34", "LOCAL", "Document generated"),
        ("19:01:40", "BLOCKED", "External endpoint request"),
    ]

    for timestamp, source, action in logs:
        cls = "bad" if source == "BLOCKED" else "good"
        st.markdown(
            f"""
            <div class="card" style="padding:11px 13px">
                <span class="muted">{timestamp}</span>
                <span style="margin-left:20px" class="{cls}">{source}</span>
                <span style="margin-left:20px">{action}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# SYSTEM MONITOR
# ============================================================

elif selected_page == "System Monitor":
    header(
        "LOCAL COMPUTE",
        "System",
        "Monitor",
        "Monitor the local hardware resources powering your sovereign AI stack.",
    )

    c1, c2, c3, c4 = st.columns(4)

    for col, name, usage, detail in [
        (c1, "CPU", "23%", "4 cores"),
        (c2, "GPU", "38%", "12 GB VRAM"),
        (c3, "RAM", "41%", "32 GB"),
        (c4, "DISK", "62%", "512 GB"),
    ]:
        with col:
            metric_card(name, usage, detail)

    section("LOCAL MODEL RUNTIME")

    for model in ["Qwen2.5-7B", "Llama 3.2 Vision", "Mistral 7B"]:
        st.markdown(
            f"""
            <div class="card" style="padding:12px 14px">
                ◈ <b>{model}</b>
                <span class="good" style="float:right">● READY</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# SETTINGS
# ============================================================

elif selected_page == "Settings":
    header(
        "CONFIGURATION",
        "System",
        "Settings",
        "Configure sovereign mode, local models, sandboxing and privacy preferences.",
    )

    section("SECURITY")

    st.toggle(
        "Sovereign Mode",
        value=True,
        help="Keep AI processing inside the local environment.",
    )
    st.toggle("Block External Network", value=True)
    st.toggle("Sandbox Code Execution", value=True)

    section("AI MODELS")

    st.selectbox(
        "Default Local Model",
        ["Qwen2.5-7B", "Llama 3.2", "Mistral 7B"],
    )

    section("SYSTEM")

    st.slider("GPU Memory Limit", 1, 24, 12)

    st.success("Configuration is stored locally.")

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-line">
        <span>Sovereign AI Workbench v1.0.0</span>
        <span style="float:right">SIH 2026 · ALL DATA PROCESSED LOCALLY</span>
    </div>
    """,
    unsafe_allow_html=True,
)