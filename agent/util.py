# Update graph.html to show the reward counters
def update_graph_html(wandb_url):
    import os

    if not os.path.exists("graph.html"):
        return

    with open("graph.html", "r") as f:
        html = f.read()

    # Find the iframe and replace the src with the wandb URL
    iframe_start = html.find("<iframe")
    iframe_end = html.find("</iframe>")
    iframe = html[iframe_start:iframe_end]

    src_start = iframe.find("src=")
    src_end = iframe.find(" ", src_start)
    src = iframe[src_start:src_end]

    src = src.replace("src=", "").replace('"', "").replace("'", "")
    html = html.replace(src, f'{wandb_url}')

    with open("graph.html", "w") as f:
        f.write(html)