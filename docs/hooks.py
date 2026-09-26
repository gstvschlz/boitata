import re

CHAPTER = re.compile(r"\]\(\.\./((\d\d)-[\w-]+)/README\.md\)")
SOURCE = re.compile(r"\]\(\.\./common\.py\)")


def on_page_markdown(markdown, page, **kwargs):
    """Points links between example chapters at their gallery pages."""
    markdown = CHAPTER.sub(r"](../\1/example_\2.md)", markdown)
    return SOURCE.sub("](https://github.com/gstvschlz/ceres/blob/main/examples/common.py)", markdown)
