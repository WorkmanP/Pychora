"""The entry point for a program that a user to input their chess opening repertoire and the
program will output similar, useful openings that the user has not utilised."""

import subprocess
import sys
import threading
import time
from os import getcwd
from os.path import exists
from typing import Any, Dict, List, Union

import PySimpleGUI as sg

from src.pychora_db import process_database
from src.pychora_rec import process_clustering

DEFAULT_CONFIG: Dict[str, Union[int, float]] = {
    "DATABASE_MAX_OPENINGS_PROCCESSED": 10_000_000,
    "DATABASE_OPENING_MIN_PROPORTION": 0.000_030,
    "DATABASE_SAMPLE_SIZE": 200_000,

    "OPENING_MOVES_PER_PLAYER": 6,
    "OPENING_LENGTH": 12,

    "THREAD_COUNT": 8
}


def main() -> int:
    window = generate_gui()
    running_threads: List[threading.Thread] = []
    while True:
        event, values = window.read()  # type: ignore

        if event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
            break

        else:
            if event == '-CLUST_SUBMIT-':
                print("--- Recommend openings chosen ---")
                process_cluster(values)

            if event == '-CONV_SUBMIT-':
                print("--- Convert database chosen ---")
                if running_threads:
                    sg.popup("Cannot convert PGN databases simultaniously")
                else:
                    t = process_conversion(values)
                    if t:
                        running_threads.append(t)

        temp_threads = []

        for thread in running_threads:
            if thread.is_alive():
                temp_threads.append(thread)

        running_threads = temp_threads.copy()

    window.close()
    return 0


def generate_gui() -> sg.Window:
    sg.theme('LightGray1')

    header: sg.Text = sg.Text(
        "Python-Chess opening recommendation algorithm",
        font=("Helvetica", 16, "bold"))

    # All GUI assets to handle user opening repertoire

    user_rep_header: List[sg.Text] = [sg.Text(
        "The following is the user's opening database to be expanded and receive recomendations towards",
        font=("Helvetica", 11, "bold"))
    ]

    user_rep_body: List = [
        sg.Text("User's game opening database in .json format: "),
        sg.Stretch(),
        sg.Input(key='-DB_OPENING_USER-'),
        sg.FileBrowse()
    ]

    # All GUI Assets to handle the large database to perform clustering analysis on
    clust_db_header: List[sg.Text] = [sg.Text(
        "The following is the large opening database to be analyse and create clusters from",
        font=("Helvetica", 11, "bold"))
    ]

    clust_db_body: List = [
        sg.Text("Large game opening database in .json format: "),
        sg.Stretch(),
        sg.Input(key='-DB_OPENING_LARGE-'),
        sg.FileBrowse()
    ]

    clust_rec_export_data: List[Any] = [
        sg.Text("Recommended openings export location: "),
        sg.Stretch(),
        sg.Input(key='-REC_DEST-'),
        sg.FolderBrowse()
    ]

    clust_visualise_data: List[Any] = [
        sg.Checkbox("Export Cluster Graphs?", key='-EXPORT-'),
        sg.Stretch(),
        sg.Text("Graph export location: "),
        sg.Input(key='-PLOT_DEST-'),
        sg.FolderBrowse()
    ]

    clust_manual_amnt_clust: List[Any] = [
        sg.Checkbox("Manually set the amount of clusters?", key='-MANUAL-'),
        sg.Stretch(),
        sg.Text("Amount of clusters: "),
        sg.Input(key='-CLUST_QUANT-', size=(5, 1))
    ]

    clust_submit: List[Any] = centre([
        sg.Submit(button_text="Recommend Openings", key='-CLUST_SUBMIT-')
    ])

    horizontal_line_1: List = centre([
        sg.Text("---------------------------------------------------------------")
    ])

    # All GUI assets for converting databases to opening types:
    conversion_header: List[sg.Text] = centre([sg.Text(
        "Convert PGN database to a JSON opening database",
        font=("Helvetica", 14, "bold"))
    ])

    # All GUI assets to handle cluster database variables
    database_var_header: List[sg.Text] = [sg.Text(
        "The following options are used during the processing of the large standard chess game database",
        font=("Helvetica", 11, "bold"))]

    database_open_len_body: List = [
        sg.Text("The amount of moves per player that constitues an opening:"),
        sg.Stretch(),
        sg.Input(key='-CONV_OPEN_LEN-', size=(10, 1))]

    database_max_amnt_body: List = [
        sg.Text("The maximum amount of games to process:"),
        sg.Stretch(),
        sg.Input(key='-CONV_MAX_PROCESS-', size=(10, 1))]

    database_open_min_prop: List = [
        sg.Text(
            "The miniumum frequency of an each recorded opening  (0 Returns all openings. Recommended 40-200):"),
        sg.Stretch(),
        sg.Input(key='-CONV_MIN_PROP-', size=(5, 1)),
        sg.Text("per 1,000,000 games.")]

    conversion_file_input: List = [
        sg.Text("A chess game database in PGN format to convert into a .json opening database:"),
        sg.Stretch(),
        sg.Input(key='-CONV_FILE_LOC-'),
        sg.FileBrowse()
    ]

    conversion_file_output: List = [
        sg.Text("The directory to place the created JSON opening database:"),
        sg.Stretch(),
        sg.Input(key='-CONV_FILE_DEST-'),
        sg.FolderBrowse()
    ]

    conv_submit: List[Any] = centre([
        sg.Submit(button_text="Convert Database", key='-CONV_SUBMIT-')
    ])

    # Output Column:
    output_header = [sg.Text("Command Outputs:",
                             font=("Helvetica", 11, "bold"))]

    output_body = [sg.Output(size=[50, 38])]

    column_1 = [user_rep_header,
                user_rep_body,
                [sg.T("")],
                clust_db_header,
                clust_db_body,
                [sg.T("")],
                clust_rec_export_data,
                clust_visualise_data,
                clust_manual_amnt_clust,
                clust_submit,
                [sg.T("")],
                horizontal_line_1,
                conversion_header,
                [sg.T("")],
                database_var_header,
                database_max_amnt_body,
                database_open_min_prop,
                database_open_len_body,
                [sg.T("")],
                conversion_file_input,
                conversion_file_output,
                conv_submit]

    column_2 = [output_header,
                output_body]

    layout = [centre([header]),
              [sg.T("")],
              [sg.Column(column_1),
              sg.Column(column_2)]]
    #   [user_rep_header],
    #   [user_rep_body],
    #   [sg.T("")],
    #   [clust_db_header],
    #   [clust_db_body],
    #   [sg.T("")],
    #   [clust_rec_export_data],
    #   [clust_visualise_data],
    #   [clust_manual_amnt_clust],
    #   [clust_submit],
    #   [sg.T("")],
    #   [horizontal_line_1],
    #   [conversion_header],
    #   [sg.T("")],
    #   [database_var_header],
    #   [database_max_amnt_body],
    #   [database_open_min_prop],
    #   [database_open_len_body],
    #   [sg.T("")],
    #   [conversion_file_input],
    #   [conversion_file_output],
    #   [conv_submit],
    #   [sg.T("")],
    # ]

    window = sg.Window('PyChora: Python-Chess opening recommendation algorithm',
                       layout, finalize=True)
    return window


def generate_output_window() -> sg.Window:
    gui_header: List[sg.Text] = [sg.Text("Current process real-time shell outputs")]
    gui_output: List[sg.Output] = [sg.Output(size=(60, 15))]

    layout: List[List[sg.Any]] = [[gui_header],
                                  [gui_output]]
    output_window: sg.Window = sg.Window(
        'PyChora: Python-Chess opening recommendation algorithm', layout, finalize=True)
    return output_window


def centre(elems):
    return [sg.Stretch(), *elems, sg.Stretch()]


def process_cluster(values: Dict[str, str]):
    if not validate_clust_inputs(values):
        sg.popup("Info: Conversion not completed", title='PyChora: Info!')
        return None

    large_db = values['-DB_OPENING_LARGE-']
    user_db = values['-DB_OPENING_USER-']

    rec_dest = values['-REC_DEST-']
    if values['-EXPORT-']:
        plt_dest = values['-PLOT_DEST-']
    else:
        plt_dest = None

    if values['-MANUAL-']:
        clust_count = int(values['-CLUST_QUANT-'])
    else:
        clust_count = 0

    thread = threading.Thread(target=process_clustering, args=(large_db,
                                                               user_db,
                                                               None,
                                                               rec_dest),
                              kwargs={'export_plots': plt_dest,
                                      'clust_count': clust_count})
    thread.start()
    return thread


def validate_clust_inputs(values: Dict[str, str]) -> bool:
    if not values['-DB_OPENING_USER-']:
        sg.popup("Error: No user opening database was supplied!", title='PyChora: Error!')
        return False

    if not values['-DB_OPENING_LARGE-']:
        sg.popup("Error: No large opening database was supplied!", title='PyChora: Error!')
        return False

    if not exists(values['-DB_OPENING_USER-']):
        sg.popup("Error: User opening file was not fount!", title='PyChora: Error!')
        return False

    if not exists(values['-DB_OPENING_LARGE-']):
        sg.popup("Error: Large opening database was not fount!", title='PyChora: Error!')
        return False

    if not values['-REC_DEST-']:
        sg.popup("Error: Recommended openings export directory not given")
        return False

    if not exists(values['-REC_DEST-']):
        sg.popup("Error: Recommended openings export directory not found")
        return False

    if values['-EXPORT-'] and not exists(values['-PLOT_DEST-']):
        sg.popup("Error: Graph export location was not found!", title='PyChora: Error!')
        return False

    if values['-MANUAL-']:
        if values['-CLUST_QUANT-'] is None or values['-CLUST_QUANT-'] == '':
            sg.popup("Error: Manual number of clusters selected but was not supplied!")
            return False
        if not values['-CLUST_QUANT-'].isdigit():
            sg.popup("Error: Manual number of clusters was not valid!", title='PyChora: Error!')
            return False

    return True


def process_conversion(values: Dict[str, str]) -> Union[threading.Thread, None]:

    if not validate_conv_inputs(values):
        sg.popup("Info: Conversion not completed", title='PyChora: Info!')
        return None
    else:
        thread = threading.Thread(target=process_database,
                                  args=(values['-CONV_FILE_LOC-'],
                                        values['-CONV_FILE_DEST-'],
                                        (int(values['-CONV_MIN_PROP-'])) / 1_000_000,
                                        int(values['-CONV_OPEN_LEN-']),
                                        int(values['-CONV_MAX_PROCESS-'])))
        thread.start()
    return thread


def create_database_command(pgn_db, dest, prop, leng, max):
    """Create a pychora_db command for the given variables"""
    cmd = 'python '
    cmd = cmd + getcwd() + '/src/pychora_db.py '
    cmd = cmd + f'{pgn_db} -d {dest} -p {prop} -l {leng} -g {max}'
    return cmd


def run_command(cmd, window: sg.Window):
    print(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in process.stdout:  # type: ignore
        line = line.decode(encoding=sys.stdout.encoding,
                           errors='replace' if (sys.version_info) < (3, 5)
                           else 'backslashreplace').rstrip()
        print(line)
        window.Refresh()
    return


def validate_conv_inputs(values: Dict[str, str]) -> bool:
    if not values['-CONV_FILE_LOC-']:
        sg.popup("Error: No game database was supplied!", title='PyChora: Error!')
        return False
    if not values['-CONV_FILE_DEST-']:
        sg.popup("Error: No output directory was supplied!", title='PyChora: Error!')
        return False
    if not values['-CONV_OPEN_LEN-']:
        sg.popup("Error: No moves for each opening was supplied!", title='PyChora: Error!')
        return False
    if not values['-CONV_MAX_PROCESS-']:
        sg.popup("Error: No maximum games to process was supplied!", title='PyChora: Error!')
        return False
    if not values['-CONV_MIN_PROP-']:
        sg.popup("Error: No minimum opening proportion was supplied!", title='PyChora: Error!')
        return False

    if not exists(values['-CONV_FILE_DEST-']):
        sg.popup("Error: Output directory does not exist!", title='PyChora: Error!')
        return False
    if not exists(values['-CONV_FILE_LOC-']):
        sg.popup("Error: Input file does not exist!", title='PyChora: Error!')
        return False

    if not values['-CONV_OPEN_LEN-'].isdigit():
        sg.popup("Error: Number of moves for each opening was not valid!", title='PyChora: Error!')
        return False
    if not values['-CONV_MAX_PROCESS-'].isdigit():
        sg.popup("Error: Maximum games to process was not valid!", title='PyChora: Error!')
        return False
    if not values['-CONV_MIN_PROP-'].isdigit():
        sg.popup("Error: Minimum opening proportion was not valid!", title='PyChora: Error!')
        return False
    return True


if __name__ == "__main__":
    main()
