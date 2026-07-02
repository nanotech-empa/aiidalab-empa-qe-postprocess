import ipywidgets as ipw

ICONS = {
    "unfold": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <path d="M10 48h44M10 16v32" />
            <path d="M14 41c8-18 15-18 23 0s12 8 17-12" />
            <path d="M17 26h7M20 23v7" />
            <path d="M37 21h14" />
            <path d="M44 14v14" opacity=".55" />
        </svg>
    """,
    "search": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <circle cx="27" cy="27" r="14" />
            <path d="M38 38l13 13" />
            <path d="M20 27h14M27 20v14" opacity=".65" />
        </svg>
    """,
}

ITEMS = [
    ("unfold", "Submit band unfolding", "Prepare and submit folded-kpoints QE calculations for BandUPpy.", "submit_qe_unfolding.ipynb"),
    ("search", "Search post-processing", "Find completed post-processing calculations and open unfolded-band viewers.", "search.ipynb"),
]


def _card(appbase, icon, title, description, notebook):
    return f"""
        <a class="qepp-card" href="{appbase}/{notebook}" target="_blank">
            <span class="qepp-icon qepp-icon-{icon}">{ICONS[icon]}</span>
            <span class="qepp-card-text">
                <span class="qepp-card-title">{title}</span>
                <span class="qepp-card-description">{description}</span>
            </span>
        </a>
    """


def get_start_widget(appbase, jupbase):  # noqa: ARG001
    cards = "".join(_card(appbase, *item) for item in ITEMS)
    return ipw.HTML(
        f"""
        <style>
            .qepp-launcher {{
                --ink: #1f2933;
                --muted: #5b6673;
                --line: #d8dee6;
                --panel: #ffffff;
                --hover: #f4f8fb;
                --blue: #0c7bb3;
                --green: #2a8c55;
                --orange: #c66a1f;
                color: var(--ink);
                max-width: 940px;
                margin: 0 auto;
                padding: 14px 6px 22px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .qepp-launcher h1 {{ font-size: 26px; font-weight: 650; margin: 0 0 8px; }}
            .qepp-launcher p {{ color: var(--muted); margin: 0 0 18px; max-width: 720px; }}
            .qepp-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 10px; }}
            .qepp-card {{
                display: grid;
                grid-template-columns: 54px minmax(0, 1fr);
                gap: 12px;
                min-height: 88px;
                padding: 12px;
                border: 1px solid var(--line);
                border-radius: 7px;
                background: var(--panel);
                color: inherit;
                text-decoration: none;
                box-sizing: border-box;
            }}
            .qepp-card:hover, .qepp-card:focus {{ background: var(--hover); border-color: #98b7cc; color: inherit; text-decoration: none; }}
            .qepp-icon {{
                width: 52px; height: 52px; border-radius: 7px;
                display: inline-flex; align-items: center; justify-content: center;
                background: #edf4f8; color: var(--blue);
            }}
            .qepp-icon svg {{ width: 42px; height: 42px; fill: none; stroke: currentColor; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}
            .qepp-icon-search {{ color: var(--green); background: #eef7f1; }}
            .qepp-card-text {{ display: flex; flex-direction: column; justify-content: center; min-width: 0; }}
            .qepp-card-title {{ font-size: 15px; font-weight: 650; line-height: 1.2; margin-bottom: 5px; }}
            .qepp-card-description {{ font-size: 13px; line-height: 1.35; color: var(--muted); }}
        </style>
        <div class="qepp-launcher">
            <h1>QE post-processing</h1>
            <p>Small notebook tools for Quantum ESPRESSO post-processing workflows. The first workflow is BandUPpy band unfolding.</p>
            <div class="qepp-grid">{cards}</div>
        </div>
        """
    )
