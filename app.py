import panel as pn
import hvplot.pandas
import pandas as pd
import numpy as np
from bokeh.models import HoverTool

pn.extension(design="material")

datadict = dict(
    x1=[1],
    x2=[4],
    img=[
        "./assets/acp000dknt.mp4",
    ],
)

hover = HoverTool(
    tooltips="""
    <div>
        <video controls autoplay src="@img" type="video/mp4" width=360 height=202></video>
    <div>
"""
)

df_test = pd.DataFrame.from_dict(datadict)
plot = df_test.hvplot.scatter(
    title="Test",
    x="x1",
    y="x2",
    s=150,
    hover_cols=["img"],
    tools=[hover],
    align="center",
    height=800,
    width=1600,
    grid=True,
)
plot_pane = pn.panel(plot)
plot_pane.servable()


"""
datadict = dict(
    x=[1, 5, 3, 1],
    y=[4, 10, 60, 5],
    img=[
        "https://user-images.githubusercontent.com/43569179/163564388-531c34f7-8ac0-4620-a2ff-3d2dc34fa324.jpg",
        "https://user-images.githubusercontent.com/43569179/163564388-531c34f7-8ac0-4620-a2ff-3d2dc34fa324.jpg",
        "assets/0FhliLmQri_frame_1.jpg",
        "assets/output.gif",
    ],
)

df = pd.read_csv(
    "/home/dl18206/Desktop/phd/code/personal/panaf-models/src/supervised/modules/camera_reaction/binary/analysis/interactive_tsne.csv"
)

df["filename"] = df.filename.str.split("/").str[-1]

plot = df.hvplot.scatter(
    x="x1",
    y="x2",
    by="label",
    hover_cols=["label", "filename"],
    legend="top",
    height=500,
    width=1000,
    align="center",
)
plot0_pane = pn.panel(plot0)

"""
