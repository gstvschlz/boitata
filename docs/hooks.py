import html
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
CHAPTERS = "01-samples-and-support 02-describing-data 03-spatial-continuity 04-kriging 05-simulation 06-checking-a-model"
LEARN_PAGE = re.compile(r"^learn/(\d\d)-[\w-]+/learn_\d\d\.md$")
HEADING = re.compile(r"^# .*$", re.MULTILINE)
DOC_TITLE = re.compile(r'^\s*(?:"""|\'\'\')\s*#\s*(.+)')
TERM = re.compile(r"^\*\[(.+?)\]:\s*(.+?)\s*$", re.MULTILINE)
CODE = re.compile(r"`([^`]+)`")
CODE_HTML = r"<code>\1</code>"
CHAPTER = re.compile(r"\]\(((?:\.\./)+(\d\d)-[\w-]+/(\d\d)-[\w-]+)/README\.md\)")
SOURCE = re.compile(r"\]\((?:\.\./)+common\.py\)")
SENTENCE = re.compile(r"(?<=\.)\s+(?=[A-Z`])")
FENCE = "```{.shell .mkd-glr-script-out-disp }"
OUTPUT = re.compile(re.escape(FENCE) + r"\n(.*?)```", re.DOTALL)
COLAB = (
    "[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/gstvschlz/boitata/blob/main/notebooks/learn_{}.ipynb)"
)
TQDM = re.compile(r"(\d+)/(\d+) \[(\d+:\d\d)|(\d+)it \[(\d+:\d\d)")

API = """
containers: containers and I/O
    Table PointSet BlockModel BlockModelFile Mesh MeshReport Polylines convex_hull grid_surface
    read_csv write_csv read_gslib write_gslib read_parquet write_parquet read_mesh write_mesh
    read_segy write_segy read_shapefile write_shapefile read_geopackage write_geopackage
    read_geotiff write_geotiff
datasets: datasets and plots
    datasets plot plot3d
drillholes: drill holes and domains
    Drillholes merge_intervals check_drillholes fix_drillholes duplicates hole_distance
    PolygonSelector point_in_polygon polygon_distance assign_domain block_shell contact_distance
    buffer_domains remove_small_units smooth_classes map_blocks domain_change transition_matrix
    vertical_proportions combine_proportions planned_drillholes snap_to_surface
eda: exploratory analysis
    Categories describe describe_by correlation swath contact soft_boundary capping capping_report
    despike data_spacing spatial_bootstrap paired_bias
transforms: transforms
    NormalScore Capping HermiteAnamorphosis BoxCox PPMT PCA MAF StepwiseConditional GaussianImputer
    Declustering cell_declustering polygon_declustering weight_declustering Trend detrend Unfold
    correct_distribution affine_correction indirect_lognormal_correction normal_cdf normal_ppf
    GaussianMixture KernelDensity
compositional: compositional data
    closure alr alr_inverse clr clr_inverse ilr ilr_inverse aitchison_distance
variography: variography
    Structure Variogram ExperimentalVariogram experimental_variogram dissemination
    VariogramSet experimental_variograms VariogramMap variogram_map h_scatter pairs
    Coregionalization Transiogram experimental_transiogram VariogramVolume variogram_volume
    local_variogram_parameters LocalAnisotropy
estimation: estimation
    Search calibrate_search neighborhood_stats HighGrade OrdinaryKriging SimpleKriging BlockKriging
    UniversalKriging ExternalDriftKriging FactorialKriging DualKriging BayesianKriging Cokriging
    IndicatorKriging MultipleIndicatorKriging CategoricalIndicatorKriging MultigaussianKriging
    InverseDistance NearestNeighbor MovingAverage MovingMedian LocalLeastSquares
recoverable: recoverable resources
    UniformConditioning DisjunctiveKriging change_of_support upscale downscale
simulation: simulation
    SGS DSS TurningBands MultivariateSimulation SIS Plurigaussian SimulationSummary CategoricalSummary
    IndicatorSummary CategoricalIndicatorSummary check_realizations RealizationCheck
    gibbs localize object_training_image training_image_consistency SNESIM ImageQuilting
    select_realizations
modeling: modeling
    ImplicitModel
validation: checking models
    CrossValidation IndicatorCrossValidation CategoricalCrossValidation validate_model global_bias
    block_correlation compare_models grade_tonnage classify uncertainty_curve required_spacing
    spacing_study
planning: drilling plans
    DrillholePlan DrillholeSearch Greedy Swap ModifiedRandomSearch Annealing
errors: errors
    BoitataError InvalidInput MissingColumn FileError
"""


def sections():
    """(slug, title, names) per API section, in `API` order."""
    for block in re.split(r"\n(?=\S)", API.strip()):
        head, _, names = block.partition("\n")
        slug, _, title = head.partition(": ")
        yield slug, title, names.split()


def on_config(config, **kwargs):
    """Builds the API nav: an overview page and one page per object in each section."""
    nav = [
        {title: [{"overview": f"api/{slug}/index.md"}] + [{name: f"api/{slug}/{name}.md"} for name in names]}
        for slug, title, names in sections()
    ]
    config["nav"] = [
        {"API": nav} if isinstance(item, dict) and "API" in item else item for item in config["nav"]
    ]
    return config


def on_files(files, config, **kwargs):
    """Generates the API pages, with summaries from the first docstring sentence and "Used in" chips, and the glossary."""
    import boitata
    from mkdocs.structure.files import File

    uses = used_in([name for _, _, names in sections() for name in names], script_pages(ROOT))
    for slug, title, names in sections():
        rows = ["| name | summary | used in |", "| --- | --- | --- |"]
        for name in names:
            doc = inspect.getdoc(getattr(boitata, name)) or ""
            summary = SENTENCE.split(" ".join(doc.split()), 1)[0].replace("|", "\\|")
            rows.append(f"| [`{name}`]({name}.md) | {summary} | {len(uses[name]) or ''} |")
            page = f"::: boitata.{name}\n    options:\n      heading_level: 1\n" + used_in_chips(uses[name])
            files.append(File.generated(config, f"api/{slug}/{name}.md", content=page))
        index = f"# {title}\n\n" + "\n".join(rows) + "\n"
        files.append(File.generated(config, f"api/{slug}/index.md", content=index))
    terms = [
        t
        for path in sorted((ROOT / "docs/snippets/glossary").glob("*.md"))
        for t in TERM.findall(path.read_text("utf-8"))
    ]
    files.append(File.generated(config, "guide/glossary.md", content=glossary(terms)))
    return files


def on_page_markdown(markdown, page, **kwargs):
    """Points links between example pages at their gallery pages, draws captured progress bars, adds the Learn trail
    and tunes search."""
    uri = page.file.src_uri
    if uri.endswith("mg_execution_times.md"):
        page.meta["search"] = {"exclude": True}
    elif uri.startswith(("learn/", "guide/")):
        page.meta["search"] = {"boost": 2}
    if match := LEARN_PAGE.match(uri):
        markdown = HEADING.sub(
            lambda h: f"{h[0]}\n\n{COLAB.format(match[1])}\n\n{trail(chapters(ROOT), match[1])}", markdown, count=1
        )
    markdown = OUTPUT.sub(progress_bars, markdown)
    markdown = CHAPTER.sub(r"](\1/example_\2_\3.md)", markdown)
    return SOURCE.sub("](https://github.com/gstvschlz/boitata/blob/main/examples/common.py)", markdown)


def chapters(root):
    """(slug, title, has page) per Learn chapter; title from its README.txt, else from the slug."""
    for slug in CHAPTERS.split():
        readme = root / "learn" / slug / "README.txt"
        head = readme.read_text("utf-8").strip().splitlines()[0] if readme.is_file() else ""
        title = head.lstrip("# ").strip() or slug[3:].replace("-", " ").capitalize()
        yield slug, title, (root / "learn" / slug / f"learn_{slug[:2]}.py").is_file()


def trail(chapters, number):
    """The chapter trail: earlier chapters done, the current one here, later ones plain; missing pages unlinked."""
    items = []
    for slug, title, exists in chapters:
        state = ' class="done"' if slug[:2] < number else ' class="here"' if slug[:2] == number else ""
        label = html.escape(f"{int(slug[:2])}. {title}")
        link = (
            f"[{label}](../{slug}/learn_{slug[:2]}.md)"
            if exists and "here" not in state
            else f"<span>{label}</span>"
        )
        items.append(f'<li{state} markdown="span">{link}</li>')
    return '<ol class="bt-trail" markdown="1">\n' + "\n".join(items) + "\n</ol>\n"


def script_pages(root):
    """(page path under docs/, title, source) for each Learn and example script: Learn, then Workflows, then the rest."""
    scripts = sorted(root.glob("learn/*/learn_*.py"))
    workflows = sorted(root.glob("examples/14-*/*/example_*.py"))
    scripts += workflows + [p for p in sorted(root.glob("examples/*/*/example_*.py")) if p not in workflows]
    for path in scripts:
        text = path.read_text("utf-8")
        title = DOC_TITLE.match(text)
        yield (
            path.relative_to(root).with_suffix(".md").as_posix(),
            title[1].strip() if title else path.stem,
            text,
        )


def used_in(names, pages):
    """Pages using each name, as `bt.Name` or, for classes, `.Name(`."""
    pages = list(pages)
    uses = {}
    for name in names:
        pattern = rf"\b(?:bt|boitata)\.{name}\b" + (rf"|\.{name}\(" if name[0].isupper() else "")
        uses[name] = [(uri, title) for uri, title, text in pages if re.search(pattern, text)]
    return uses


def used_in_chips(links, cap=8):
    """The "Used in" chips linking an API page (two levels deep) to the first `cap` gallery pages."""
    if not links:
        return ""
    chips = "\n".join(
        f'<li markdown="span">[{html.escape(title)}](../../{uri})</li>' for uri, title in links[:cap]
    )
    return f'\n**used in**\n\n<ul class="bt-used-in" markdown="1">\n{chips}\n</ul>\n'


def glossary(terms):
    """The glossary page as one raw HTML definition list, so the abbreviation tooltips leave its terms alone."""
    unique = {term.casefold(): (term, text) for term, text in reversed(terms)}
    rows = "".join(
        f'<dt id="{re.sub(r"[^a-z0-9]+", "-", key).strip("-")}">{html.escape(term)}</dt>'
        f"<dd>{CODE.sub(CODE_HTML, html.escape(text))}</dd>\n"
        for key, (term, text) in sorted(unique.items())
    )
    intro = "short definitions of the terms these pages use. elsewhere, hover a dotted term to see its definition."
    return f"# glossary\n\n{intro}\n\n<dl>\n{rows}</dl>\n"


def progress_bars(block):
    """Replaces each captured tqdm bar, whose states arrive one per line, with its last state drawn filling."""
    parts, text, bar = [], [], None

    def flush():
        nonlocal bar
        if bar:
            done, total, elapsed, count, count_elapsed = bar
            label = f"{done}/{total}" if total else count
            share = min(100, 100 * int(done) // max(int(total), 1)) if total else 100
            parts.append(
                f'<div class="bt-progress" style="--share: {share}%"><span class="bt-progress-track">'
                f'<span class="bt-progress-fill"></span></span><code>{label} · {elapsed or count_elapsed}</code></div>'
            )
            bar = None
        if "".join(text).strip():
            parts.append(FENCE + "\n" + "\n".join(text).strip("\n") + "\n```")
        text.clear()

    for line in block[1].split("\n"):
        state = TQDM.findall(line)
        if not state:
            if bar and line.strip():
                flush()
            text.append(line)
            continue
        if (state[-1][0] or state[-1][3]) == "0" or text and "".join(text).strip():
            flush()
        text.clear()
        bar = state[-1]
    flush()
    return "\n\n".join(parts)
