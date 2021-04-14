from pathlib import Path
OUTPUT_FOLDER = 'outputs'
LOG_FOLDER = OUTPUT_FOLDER + '/logs'
FIG_FOLDER = OUTPUT_FOLDER + '/figs'
HIST_FOLDER = OUTPUT_FOLDER + '/hists'

def setup_env():
    """
    create folders for output logs/figs/hists
    logs: logs of the simulation
    figs: graph files
    hists: .pickle of simulation history
    """

    Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(FIG_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(HIST_FOLDER).mkdir(parents=True, exist_ok=True)