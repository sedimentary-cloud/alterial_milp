from .solution import *  # noqa: F401,F403

__all__ = ["ViolationRecord", "SolutionRecord", "SolutionDecoder", "ReportBuilder",
           "TimeSpaceDiagram", "ParetoFrontierDiagram", "GridHeatmapDiagram"]


def __getattr__(name):
    if name in ("TimeSpaceDiagram", "ParetoFrontierDiagram", "GridHeatmapDiagram"):
        from .diagram import TimeSpaceDiagram, ParetoFrontierDiagram
        from .heatmap import GridHeatmapDiagram
        return {"TimeSpaceDiagram": TimeSpaceDiagram,
                "ParetoFrontierDiagram": ParetoFrontierDiagram,
                "GridHeatmapDiagram": GridHeatmapDiagram}[name]
    raise AttributeError(name)
