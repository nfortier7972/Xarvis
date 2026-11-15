"""The start window: pick a known object from a drop-down list, or point at any sky coordinates.

The list holds every target analysed so far (results folder + registry, see results.known_objects), plus the
standard test objects. If the results folder is missing, it is regenerated first from the known objects
(results.rebuild_results - quick while their images are still in the disk cache).
"""

import threading
import tkinter as tk
from tkinter import ttk

from .results import DEFAULT_OBJECTS, known_objects, rebuild_results, results_dir

MODE_TEXT = {'asteroid': 'asteroid', 'named star': 'star', 'coordinates': 'sky position', 'star': 'star (snapped)',
             'catalog star': 'catalog star', 'path': 'new moving object'}


def _entries():
    """(label, settings) for the drop-down: analysed targets first (most recent first), then untried defaults."""
    def same(a, b):         # the same target, however it was written ('eps Ind B' / 'Epsilon Indi B' via SIMBAD id)
        key = lambda s: (s.get('mode'), str(s.get('star') or s.get('asteroid') or s.get('coords')).lower().replace(' ', ''))
        return key(a) == key(b)

    entries, names = [], set()
    for obj in known_objects():
        mode = MODE_TEXT.get(obj['settings'].get('mode'), obj['settings'].get('mode', '?'))
        frames = f", {obj['frames']} frames" if obj.get('frames') else ''
        when = f", {obj['saved'][:10]}" if obj.get('saved') else ''
        entries.append((f"{obj['name']}   ({mode}{frames}{when})", obj['settings']))
        names.add(obj['name'])
    for name, settings in DEFAULT_OBJECTS:
        if name not in names and not any(same(s, settings) for _, s in entries):
            entries.append((f"{name}   ({MODE_TEXT[settings['mode']]}, not analysed yet)", settings))
    return entries


def choose_target():
    """Show the start window; returns the chosen target's settings (plus 'max_images'), or None if closed."""
    from .main import parse_coords
    root = tk.Tk()
    root.title('SPHEREx explorer - choose a target')
    root.resizable(False, False)
    chosen = {}
    frame = ttk.Frame(root, padding=14)
    frame.grid()

    ttk.Label(frame, text='Known objects (cached results)', font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky='w')
    combo = ttk.Combobox(frame, state='readonly', width=72)
    combo.grid(row=1, column=0, sticky='we', pady=(4, 0))
    open_button = ttk.Button(frame, text='Open')
    open_button.grid(row=1, column=1, padx=(8, 0), pady=(4, 0))
    status = ttk.Label(frame, text='', foreground='#555')
    status.grid(row=2, column=0, columnspan=2, sticky='w', pady=(2, 10))

    ttk.Separator(frame).grid(row=3, column=0, columnspan=2, sticky='we', pady=6)
    ttk.Label(frame, text='Or point at any sky coordinates', font=('Segoe UI', 10, 'bold')).grid(row=4, column=0, columnspan=2, sticky='w')
    coords = ttk.Entry(frame, width=74)
    coords.grid(row=5, column=0, sticky='we', pady=(4, 0))
    go_button = ttk.Button(frame, text='Go')
    go_button.grid(row=5, column=1, padx=(8, 0), pady=(4, 0))
    ttk.Label(frame, foreground='#555', justify='left',
              text='RA Dec, e.g.  08h55m10.8s -07d14m42s   or   08:55:10.8 -07:14:42   or   133.795 -7.245 (degrees)\n'
                   'Every object in the image is identified (Hipparcos, Gaia DR3, CatWISE, planets, known asteroids and\n'
                   'comets); anything unknown is marked NEW and its coordinates + times are saved.').grid(
        row=6, column=0, columnspan=2, sticky='w', pady=(2, 10))
    error = ttk.Label(frame, text='', foreground='#b00020')
    error.grid(row=7, column=0, columnspan=2, sticky='w')

    ttk.Separator(frame).grid(row=8, column=0, columnspan=2, sticky='we', pady=6)
    options = ttk.Frame(frame)
    options.grid(row=9, column=0, columnspan=2, sticky='w')
    ttk.Label(options, text='Max images (blank = all):').grid(row=0, column=0)
    max_images = ttk.Entry(options, width=8)
    max_images.grid(row=0, column=1, padx=6)

    entries = []

    def refresh():
        entries[:] = _entries()
        combo['values'] = [label for label, _ in entries]
        if entries:
            combo.current(0)
        status.config(text=f'{len(entries)} objects - results in {results_dir()}')

    def limit():
        text = max_images.get().strip()
        if not text:
            return None
        if not text.isdigit() or int(text) < 1:
            raise ValueError('Max images must be a whole number')
        return int(text)

    def finish(settings):
        try:
            settings = dict(settings, max_images=limit())
        except ValueError as e:
            error.config(text=str(e))
            return
        chosen['settings'] = settings
        root.destroy()

    def open_selected(*_):
        if combo.current() >= 0:
            finish(entries[combo.current()][1])

    def go(*_):
        text = coords.get().strip()
        try:
            coord = parse_coords(text)
        except Exception:
            error.config(text='Could not read those coordinates - e.g. "08h55m10.8s -07d14m42s" or "133.795 -7.245"')
            return
        finish({'mode': 'coordinates', 'coords': [float(coord.ra.deg), float(coord.dec.deg)]})

    open_button.config(command=open_selected)
    go_button.config(command=go)
    coords.bind('<Return>', go)
    combo.bind('<Return>', open_selected)

    folder = results_dir()
    if not folder.exists() or not any(folder.glob('*.ecsv')):
        # no results folder: regenerate it from the known objects, in the background, then fill the list
        for widget in (combo, open_button, coords, go_button):
            widget.state(['disabled'])
        progress = {'text': 'Results folder not found - regenerating it from the known objects...', 'done': False}

        def work():
            rebuild_results(on_progress=lambda text: progress.update(text=text + ' (see the terminal)'))
            progress['done'] = True

        threading.Thread(target=work, daemon=True).start()

        def poll():
            status.config(text=progress['text'])
            if progress['done']:
                for widget in (open_button, coords, go_button):
                    widget.state(['!disabled'])
                combo.state(['!disabled', 'readonly'])
                refresh()
            else:
                root.after(500, poll)
        poll()
    else:
        refresh()

    coords.focus_set()
    root.mainloop()
    return chosen.get('settings')
