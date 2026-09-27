import re

CHAPTER = re.compile(r"\]\(((?:\.\./)+((?:tutorials|topics)/)?(\d\d)-[\w-]+)/README\.md\)")
SOURCE = re.compile(r"\]\((?:\.\./)+common\.py\)")


def on_page_markdown(markdown, page, **kwargs):
    """Points links between example pages at their gallery pages."""

    def chapter(match):
        gallery = match[2] or page.file.src_uri
        stem = "tutorial" if gallery.startswith("tutorials") or "/tutorials/" in gallery else "example"
        return f"]({match[1]}/{stem}_{match[3]}.md)"

    markdown = CHAPTER.sub(chapter, markdown)
    return SOURCE.sub("](https://github.com/gstvschlz/ceres/blob/main/examples/common.py)", markdown)
