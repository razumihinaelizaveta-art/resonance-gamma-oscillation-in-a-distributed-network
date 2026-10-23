

import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation
import matplotlib.widgets as widgets


class VisualizItem:
    plot_type_list = [ 'spatial', 'raster', 'trace']
    time_plot_list = ['raster', 'trace']
    def __init__(self, name:str, config:dict, data:dict, figure, axis, time:float, dt:float, exp_tail:bool):
        self.name       = name
        self.config     = config
        self.data       = data
        self.figure     = figure
        self.axis       = axis
        self.exp_tail   = exp_tail
        if not 'type' in config:
            raise RuntimeError(f'There is no `type` variable in `{name}` plot')

        self.type       = config['type']
        if   self.type == 'spatial':
            self._build_spatial(time,dt)
        elif self.type == 'raster':
            self._build_raster(time,dt)
        elif self.type == 'trace':
            self._build_trace(time,dt)

    def _compute_alpha(self, pop:str, time:float, dt:float):
        act_alpha  = -np.inf * np.ones(self.popsize[pop])
        for nid in range(self.popsize[pop]):
            spk = self.spiketime[pop][\
                np.where(\
                    np.logical_and(
                        self.spiketime[pop] <= time,
                        self.neuronid[pop]  == nid
                    )
                ) ]
            if   spk.shape[0] < 1:
                continue
            elif spk.shape[0] == 1:
                act_alpha[nid] = spk[0]
            else:
                act_alpha[nid] = np.max(spk)


        if self.exp_tail:
            act_alpha = np.exp((act_alpha-time)/dt)
        else:
            act_alpha = 1.0 - (time - act_alpha)/dt
        act_alpha = np.clip(act_alpha,0,1)
        return act_alpha
        
    def _build_spatial(self,time:float,dt:float):
        self.spiketime = {}
        self.neuronid  = {}
        self.popsize   = {}
        self.views     = {}
        self.tmax      = -1.0
        for pop, layer in self.config.get('spatial', {}).items():
            if not 'positions' in layer:
                raise RuntimeError(f"There is no `position` reference in population `{pop}` of spatial plot `{self.name}`")
            if not 'spikes' in layer:
                raise RuntimeError(f"There is no `spikes` reference in population `{pop}` of spatial plot `{self.name}`")
            pos                 = np.asarray(self.data[layer['positions']])
            self.popsize[pop]   = pos.shape[0]
            self.spiketime[pop] = np.asarray(self.data[layer['spikes']])[:,0].flatten()
            self.neuronid[pop]  = np.asarray(self.data[layer['spikes']])[:,1].flatten().round().astype(int)
            if self.spiketime[pop].shape[0] == 0: continue
            tmax                = np.amax(self.spiketime[pop])
            if tmax > self.tmax : self.tmax = tmax
            

            color_cfg  = layer.get('color', 'black')
            marker     = layer.get('marker', 'o')
            size       = layer.get('size', 10)
            self.alpha = layer.get('alpha', 1.0)
            act_alpha  = self._compute_alpha(pop,time,dt)
            
            if isinstance(color_cfg, str) and color_cfg in self.data:
                values = np.asarray(self.data[color_cfg]).squeeze()
                cmap = plt.get_cmap(layer.get('color_map', 'viridis'))
                self.views[pop] = \
                    self.axis.scatter(pos[:, 0], pos[:, 1],\
                    c=values, vmin=np.amin(values), vmax=np.amax(values),\
                    s=size, marker=marker,\
                    alpha=act_alpha)
            else:
                self.views[pop] = \
                    self.axis.scatter(pos[:, 0], pos[:, 1],\
                    c=color_cfg,\
                    s=size, marker=marker,\
                    alpha=act_alpha)
        self.time_indicator = self.axis.set_title(f'Current  t={time:.1f} ms')
        #self.axis.axis("off")
        self.axis.set_yticks([])
        self.axis.set_xticks([])
            
    def _build_raster(self,time:float,dt:float):
        offset     = 0
        self.tmax  = -1.0
        for name, src in self.config.get('raster', {}).items():
            spikes = np.asarray(self.data[src['spikes']])
            if spikes.shape[0] == 0: continue
            tmax   = np.amax( spikes[:,0].flatten() )
            if tmax > self.tmax : self.tmax = tmax

            color  = src.get('color', 'black')
            marker = src.get('marker', '|')
            size   = src.get('size', 4)

            if spikes.size:
                nids = spikes[:, 1].round().astype(int)
                self.axis.plot(
                    spikes[:, 0],
                    nids  + offset ,
                    marker,
                    c=color,
                    ms=size
                )
                offset += int(nids.max()) + 2
                self.axis.axhline(offset-1, color='0.8')

        self.line = self.axis.axvline(time, color='red', linewidth=1)
        self.rec  = self.axis.axvspan(time-dt, time, color='red', alpha=0.1)
        self.axis.set_title('Raster: '+self.name)

    def _build_trace(self,time:float,dt:float):
        offset_cfg = self.config.get('offset', 'auto')
        trace_cfg  = self.config.get('traces', {})

        if offset_cfg == 'auto':
            ranges = []
            for src in trace_cfg.values():
                source = src['source']
                state = src['state']
                values = self.data[f'{source}_{state}']
                ranges.append(float(np.nanmax(values) - np.nanmin(values)))
            offset_step = max(ranges) * 0.1 if ranges else 1.0
        else:
            offset_step = float(offset_cfg)

        global_offset = 0
        self.tmax     = -1.0

        for name, src in trace_cfg.items():

            source = src['source']
            state = src['state']
            color = src.get('color', 'black')

            values = self.data[f'{source}_{state}']
            t      = self.data[f'{source}_time']
            ids    = self.data[f'{source}_ids']
            tmax   = np.amax(t)
            if tmax > self.tmax : self.tmax = tmax

            selector = src.get('neurons', f'{source}_ids')
            env = dict(self.data)
            env['np'] = np
            chosen = np.asarray(eval(selector, {}, env))

            for neuron_id in chosen:
                colidx, = np.where(ids == neuron_id)
                if colidx.shape[0] == 0: continue
                if colidx.shape[0] != 1:
                    raise RuntimeError(f"Found too many traces: {colidx}, nid={neuron_id}, all ids={ids}")
                colidx = colidx[0]
                self.axis.plot(t, values[:, colidx] + global_offset, color=color, lw=0.8)
                global_offset += offset_step

        self.line = self.axis.axvline(time, color='red', linewidth=1)
        self.rec  = self.axis.axvspan(time-dt, time, color='red', alpha=0.1)
        self.axis.set_title(self.name)
        
    def update(self,time:float, tail:float):
        updated = []
        if   self.type == 'spatial':
            for pop in self.views:
                self.views[pop].set(alpha=self._compute_alpha(pop,time,tail))
                updated.append( self.views[pop] )
            self.time_indicator.set_text( f'Current  t={time:.1f} ms' )
        elif self.type == 'raster' or self.type == 'trace':
            self.rec.set_width(tail)
            self.rec.set_x(time-tail)
            self.line.set(xdata=[time,time])
            updated += [ self.line, self.rec ]
        return updated
        
class Visualizer:
    plot_type_list = VisualizItem.plot_type_list #add here new class if needed
    time_plot_list = VisualizItem.time_plot_list

    def __init__(self, config, data, export_only:bool=False, video:bool=False):
        self.config           = config
        self.data             = data
        self.video            = video

        self.playback         = config.get('playback', {})
        self.figures_cfg      = config.get('figures', {})
        self.export           = config.get('export', {})

        self.dt               = float(self.playback.get('dt', 1.0))
        self.tail_time        = float(self.playback.get('tail_time', 20.0))
        self.tail_exponential = bool(self.playback.get('tail_exponential', False))
        self.play_fps         = int(self.playback.get('fps', 10))

        self.current_time     = float(self.playback.get('t_start', 0.0))
        self.time_start       = float(self.playback.get('t_start', 0.0))
        
        

        self.figures = {}
        self.axes    = {}
        self.plots   = []
        self.play    = False
        
        time_axis = None
        for fig_name, fig_cfg in self.figures_cfg.items():
            size = fig_cfg.get('size', [10, 8])
            fig = plt.figure(fig_name, figsize=size)

            rows = fig_cfg['grid']['rows']
            cols = fig_cfg['grid']['cols']

            self.figures[fig_name] = fig
            self.axes[fig_name] = {}
            
            
            for plot_name, plot_cfg in fig_cfg.get('plots', {}).items():
                # Checking plot type
                if not 'type' in plot_cfg:
                    raise RuntimeError(f"Figure `{fig_name}` plot `{plot_name}` does not have type specification")
                if plot_cfg['type'] not in self.plot_type_list: 
                    raise RuntimeError(f"Figure `{fig_name}` plot `{plot_name}` has unsupported type `{plot_cfg['type']}'")
                # Checking plot position
                if not 'position' in plot_cfg:
                    plot_cfg['position'] = [ 0, 0]
                if not type(plot_cfg['position']) is list:
                    raise RuntimeError(f"Figure `{fig_name}` plot `{plot_name}`: position is not a list")
                if len(plot_cfg['position']) != 2:
                    raise RuntimeError(f"Figure `{fig_name}` plot `{plot_name}`: position does not have 2 numbers: x, y {plot_cfg['position']}")
                for pid,p in enumerate(plot_cfg['position']):
                    if not type(p) is int:
                        raise RuntimeError(f"Figure `{fig_name}` plot `{plot_name}`: position entrance #{pid+1} is not an integer: {p}:{type(p)}")
                
                col, row = plot_cfg['position']
                w        = plot_cfg.get('width',1)
                h        = plot_cfg.get('height',1)
                ptype    = plot_cfg['type']
                if ptype in self.time_plot_list:
                    ax = plt.subplot2grid(
                        (rows, cols),
                        (row , col ),
                        rowspan=h,
                        colspan=w,
                        fig=fig,
                        sharex = time_axis
                    ) if time_axis else plt.subplot2grid(
                        (rows, cols),
                        (row , col ),
                        rowspan=h,
                        colspan=w,
                        fig=fig,
                    )
                    self.axes[fig_name][plot_name] = (ax, fig)
                    if not time_axis:
                        time_axis = ax
                else:
                    ax = plt.subplot2grid(
                        (rows, cols),
                        (row , col ),
                        rowspan=h,
                        colspan=w,
                        fig=fig,
                    )
                    self.axes[fig_name][plot_name] = (ax, fig)
                    
                self.plots.append( VisualizItem(\
                                    plot_name, plot_cfg, data,\
                                    fig, ax,\
                                    self.current_time, self.tail_time, self.tail_exponential)
                                )
        
        # Set time_end
        self.time_end = float( self.playback.get('t_end', max([ o.tmax for o in self.plots ]) ) )
        if export_only:
            self._export()
            return
        if video:
            self._video()
            return
        if self.export.get('show', True) and not export_only:
            self._build_controls()
            timer_fig = [ fig for fig in self.figures ]
            if len(timer_fig) != 0:
                timer_fig = timer_fig[0]
                self.timer = self.figures[timer_fig].canvas.new_timer(interval=int(1000 / self.play_fps))
                self.timer.add_callback(self._advance)
                self.timer.start()
            plt.show()
    
    def _build_controls(self):
        self.controls_fig = plt.figure(figsize=(6, 2))
        ax = self.controls_fig.add_axes([0, 0, 1, 1])   # full-figure axis (hidden)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # ── Row 1 
        ROW1_Y = 0.80

        ax_btn_l = self.controls_fig.add_axes([0.0  , ROW1_Y, 0.1, 0.1])
        self.btn_left = widgets.Button(ax_btn_l, "<")
        self.btn_left.on_clicked(lambda _: self._update(-1))

        ax_slider = self.controls_fig.add_axes([0.25, ROW1_Y, 0.5, 0.1])
        self.time_slider = widgets.Slider(
            ax_slider, 
            "Time (ms)", 
            self.time_start, 
            self.time_end, 
            valinit=self.current_time)
        self.time_slider.on_changed(lambda _:self._update())
        
        ax_btn_r  = self.controls_fig.add_axes([0.9  , ROW1_Y, 0.1, 0.1])
        self.btn_right = widgets.Button(ax_btn_r, ">")
        self.btn_right.on_clicked( lambda _: self._update( 1) )

        # ── Row 2 
        ROW2_Y = 0.60
        
        ax_play  = self.controls_fig.add_axes([0.0, ROW2_Y, 0.1, 0.1])
        self.btn_play =  widgets.Button(ax_play, "=>")
        self.btn_play.on_clicked(self._toggle_play)

        ax_text  = self.controls_fig.add_axes([0.30, ROW2_Y, 0.20, 0.1])
        self.text_box = widgets.TextBox(ax_text, "Step (ms):  ", initial=f"{self.dt}")
        self.text_box.on_submit(self._set_dt)

        ax_tail  = self.controls_fig.add_axes([0.70, ROW2_Y, 0.20, 0.1])
        self.tail_box = widgets.TextBox(ax_tail, "Tail (ms):  ", initial=f"{self.tail_time}")
        self.tail_box.on_submit(self._set_tail)
        
        # ── Row 3 
        ROW3_Y = 0.40
        ax_export      = self.controls_fig.add_axes([0.0, ROW3_Y, 0.20, 0.1])
        self.btn_export = widgets.Button(ax_export, "Export Graphs")
        self.btn_export.on_clicked( self._export )
        
    def _advance(self):
        if self.play: self._update(1)

    def _toggle_play(self,event):
        self.play = not self.play
        self.btn_play.label.set_text('||' if self.play else '=>')
    
    def _set_dt(self,text):
        try:
            self.dt = float(text)
        except BaseException as e:
            print(f"Cannot convert {text} to float")
        self._update()

    def _set_tail(self,text):
        try:
            self.tail_time = float(text)
        except BaseException as e:
            print(f"Cannot convert {text} to float")
        self._update()
    
    def _update(self,direction=None):
        
        if direction:
            if direction > 0:
                self.current_time += self.dt
            elif direction < 0:
                self.current_time -= self.dt
            if not self.video:
                self.time_slider.set_val( self.current_time )
        elif not self.video:
            self.current_time = self.time_slider.val
        if self.current_time > self.time_end:
            self.current_time = self.time_end
            if self.play: self._toggle_play(None)
                
        if self.current_time < 0.0:
            self.current_time = 0.0
        
        updated = []
        for o in self.plots:
            updated += o.update(self.current_time, self.tail_time)
        
        if not self.video:
            for fig in self.figures:
                self.figures[fig].canvas.draw_idle()
                # self.figures[fig].canvas.draw()
        return updated
    # Export figures.
    def _export(self,event=None):
        prefix  = self.export.get('prefix', '')
        export  = self.export.get('figures', True)
        formats = self.export.get('formats',['svg','jpg'])
        if not export: return
        if type(formats) is str:
            if " " in formats or "," in formats:
                formats = [ y for x in formats.split() for y in x.split(',') ]
            else:
                formats = [formats]
        if type(formats) is not list:
            raise RuntimeError(f"Export formats for figures are not a list: {formats}")
        
        for frmi, frm in enumerate(formats):
            if type(frm) is not str:
                raise RuntimeError(f"Export format #{frmi+1} is not a string: {frm}")
            for fig in self.figures:
                self.figures[fig].savefig(f"{prefix}-{fig}.{frm}")
    def _video(self,event=None):
        from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
        prefix  = self.export.get('video_prefix', '')
        export  = self.export.get('video', True)
        if not export: return
        vfps    = self.export.get('video_fps', 30)
        if type(vfps) is not int:
            raise RuntimeError(f"video_fps is not an integer: {vfps}")
        vformat = self.export.get('video_format','mp4')
        if vformat == 'mp4':
            writer = FFMpegWriter(fps=vfps, bitrate=2400)
        else:
            writer = PillowWriter(fps=vfps)

        
        v_start = self.export.get('t_start', None)
        v_stop  = self.export.get('t_stop', None)
        if not v_start: v_start = self.time_start
        if not v_stop : v_stop  = self.time_end
        
        cur_time = self.current_time
        n_frames = int ( round( (v_stop - v_start)/self.dt ) )
        
        for fig in self.figures:
            self.current_time = v_start
            self._update()
            anim = FuncAnimation(self.figures[fig], lambda k: self._update(1),
                         frames=n_frames, blit=False)
            out_path = f"{prefix}-{fig}.{vformat}"
            try:
                anim.save(out_path, writer=writer, dpi=95)
                print("saved:", out_path)
            except Exception as err:
                print("SAVING FAILED:", err)
                if vformat == 'mp4':
                    print("  mp4 needs ffmpeg on your PATH - try SAVE_FORMAT='gif' instead.")
        
        self.current_time = cur_time
         


                    
                
    
if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('npz')
    parser.add_argument('-c', '--config' , default=None , type=str           , help="Read model configuration from different file")
    parser.add_argument('--export-only'  , default=False, action='store_true', help="Export figures and exit")
    parser.add_argument('--export-video' , default=False, action='store_true', help="Export Video and exit")
    args = parser.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    data = dict(npz)

    if args.config:
        try:
            with open(args.config, 'r') as fd:
                model = yaml.safe_load(fd)  
            print(f'Loaded a model configuration from the file `{args.config}`')
        except BaseException as e:
            exit(f'Cannot read model YAML file `{args.config}`: {e}')
    else:
        if 'model' not in data:
            raise RuntimeError('NPZ does not contain model YAML')

        model_yaml = str(data['model'])
        model = yaml.safe_load(model_yaml)
    
    config = model['visualize'] if 'visualize' in model else model
    visualizer = Visualizer(config, data, export_only=args.export_only, video=args.export_video)
