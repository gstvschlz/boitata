import inspect
import re

from mkdocs.structure.files import File

CHAPTER = re.compile(r"\]\(((?:\.\./)+(\d\d)-[\w-]+/(\d\d)-[\w-]+)/README\.md\)")
SOURCE = re.compile(r"\]\((?:\.\./)+common\.py\)")
SENTENCE = re.compile(r"(?<=\.)\s+(?=[A-Z`])")

API = """
containers: Containers and I/O
    Table PointSet BlockModel Mesh Polylines read_csv write_csv read_gslib write_gslib read_segy
    write_segy
datasets: Datasets and plots
    datasets plot plot3d
transforms: Transforms
    NormalScore Capping HermiteAnamorphosis BoxCox PPMT PCA MAF StepwiseConditional GaussianImputer
variography: Variography
    Structure Variogram ExperimentalVariogram experimental_variogram dissemination
    VariogramSet experimental_variograms VariogramMap variogram_map
estimation: Estimation
    Search HighGrade OrdinaryKriging SimpleKriging IndicatorKriging UniversalKriging
    ExternalDriftKriging FactorialKriging BlockKriging
simulation: Simulation
    SGS TurningBands MultivariateSimulation SIS Plurigaussian SimulationSummary CategoricalSummary
    gibbs localize object_training_image
    training_image_consistency
    SNESIM
    ImageQuilting
drillholes: Drillholes and blocks
    Drillholes merge_intervals check_drillholes fix_drillholes PolygonSelector point_in_polygon
    polygon_distance assign_domain block_shell
modeling: Modeling
    ImplicitModel
eda: EDA and validation
    Categories describe describe_by grade_tonnage swath contact soft_boundary capping capping_report
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
        {title: [{"Overview": f"api/{slug}/index.md"}] + [{name: f"api/{slug}/{name}.md"} for name in names]}
        for slug, title, names in sections()
    ]
    config["nav"] = [
        {"API": nav} if isinstance(item, dict) and "API" in item else item for item in config["nav"]
    ]
    return config


def on_files(files, config, **kwargs):
    """Generates the API pages, with summaries from the first docstring sentence."""
    import boitata

    for slug, title, names in sections():
        rows = ["| Name | Summary |", "| --- | --- |"]
        for name in names:
            doc = inspect.getdoc(getattr(boitata, name)) or ""
            summary = SENTENCE.split(" ".join(doc.split()), 1)[0].replace("|", "\\|")
            rows.append(f"| [`{name}`]({name}.md) | {summary} |")
            page = f"::: boitata.{name}\n    options:\n      heading_level: 1\n"
            files.append(File.generated(config, f"api/{slug}/{name}.md", content=page))
        index = f"# {title}\n\n" + "\n".join(rows) + "\n"
        files.append(File.generated(config, f"api/{slug}/index.md", content=index))
    return files


def on_page_markdown(markdown, page, **kwargs):
    """Points links between example pages at their gallery pages."""
    markdown = CHAPTER.sub(r"](\1/example_\2_\3.md)", markdown)
    return SOURCE.sub("](https://github.com/gstvschlz/boitata/blob/main/examples/common.py)", markdown)
