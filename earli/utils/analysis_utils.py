# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import collections
import warnings

from .safe_serialization import safe_pickle_load

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, LineString
from . import routes_matching


def quantile(x, q, w=None, is_sorted=False, estimate_underlying_quantile=False):
    n = len(x)
    x = np.array(x)
    q = np.array(q)

    # If we estimate_underlying_quantile, we refer to min(x),max(x) not as
    #  quantiles 0,1, but rather as quantiles 1/(n+1),n/(n+1) of the
    #  underlying distribution from which x is sampled.
    if estimate_underlying_quantile and n > 1:
        q = q * (n+1)/(n-1) - 1/(n-1)
        q = np.clip(q, 0, 1)

    # Unweighted quantiles
    if w is None:
        return np.percentile(x, 100*q)

    # Weighted quantiles
    x = np.array(x)
    w = np.array(w)
    if not is_sorted:
        ids = np.argsort(x)
        x = x[ids]
        w = w[ids]
    w = np.cumsum(w) - 0.5*w
    w -= w[0]
    w /= w[-1]
    return np.interp(q, w, x)

def plot_quantiles(x, ax=None, q=None, showmeans=True, means_args=None,
                   dots=False, **kwargs):
    if ax is None: ax = Axes(1,1)[0]
    if q is None: q = np.arange(101) / 100
    m = np.mean(x)
    x = quantile(x, q)
    h = ax.plot(100*q, x, '.-' if dots else '-', **kwargs)
    if showmeans:
        if means_args is None: means_args = {}
        # ax.axhline(m, linestyle='--', color=h[0].get_color(), **means_args)
        qm = q[int(np.sum(x <= m)*(len(q)-1)/len(q))]
        ax.plot([100*qm], [m], 'o', color=h[0].get_color(), **means_args)
    return ax

def qplot(data, y, x=None, hue=None, ax=None, add_sample_size=True, cmap=None, **kwargs):
    if ax is None: ax = Axes(1,1)[0]

    if hue is None:
        plot_quantiles(data[y], ax=ax, **kwargs)
        same_samp_size = True
        n = len(data)
    else:
        hue_vals = pd.unique(data[hue].values)
        same_samp_size = len(pd.unique([(data[hue]==hv).sum()
                                        for hv in hue_vals])) == 1
        n = int(len(data) // len(hue_vals))
        for i, hv in enumerate(hue_vals):
            d = data[data[hue]==hv]
            lab = hv
            if add_sample_size and not same_samp_size:
                lab = f'{lab} (n={len(d):d})'
            if cmap is None:
                plot_quantiles(d[y], ax=ax, label=lab, **kwargs)
            else:
                c = plt.get_cmap(cmap)(i/(len(hue_vals)-1))
                plot_quantiles(d[y], ax=ax, label=lab, color=c, **kwargs)
        ax.legend(fontsize=13)

    xlab = 'quantile [%]'
    if x: xlab = f'{x} {xlab}'
    if add_sample_size and same_samp_size: xlab = f'{xlab}\n({n:d} samples)'
    labels(ax, xlab, y, fontsize=15)

    return ax

def paired_permutation_test(x, B=1000, fun=np.mean):
    n = len(x)
    perm_ref = np.zeros(B)
    for i in range(B):
        x_perm = x * ((-1)**np.random.randint(0, 2, n))
        perm_ref[i] = fun(x_perm)
    p = np.mean(fun(x) > perm_ref)
    return p

def smooth(y, n=10, deg=2):
    from scipy import signal
    n = min(n, len(y))
    if n%2 == 0: n -= 1
    return signal.savgol_filter(y, n, deg)

def labels(ax, xlab=None, ylab=None, title=None, fontsize=12):
    if isinstance(fontsize, int):
        fontsize = 3*[fontsize]
    if xlab is not None:
        ax.set_xlabel(xlab, fontsize=fontsize[0])
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=fontsize[1])
    if title is not None:
        ax.set_title(title, fontsize=fontsize[2])

class Axes:
    def __init__(self, N, W=2, axsize=(5,3.5), grid=1, fontsize=13, **subplot_args):
        self.fontsize = fontsize
        self.N = N
        self.W = W
        self.H = int(np.ceil(N/W))
        self.axs = plt.subplots(self.H, self.W, figsize=(self.W*axsize[0], self.H*axsize[1]),
                                **subplot_args)[1]
        for i in range(self.N):
            if grid == 1:
                self[i].grid(color='k', linestyle=':', linewidth=0.3)
            elif grid ==2:
                self[i].grid()
        for i in range(self.N, self.W*self.H):
            self[i].axis('off')

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        if self.H == 1 and self.W == 1:
            return self.axs
        elif self.H == 1 or self.W == 1:
            return self.axs[item]
        return self.axs[item//self.W, item % self.W]

    def labs(self, item, *args, **kwargs):
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self.fontsize
        labels(self[item], *args, **kwargs)

def tw_analysis(test_paths, positions, norm=1):
    methods = []
    problems = []
    heads = []
    travels = []
    times = []
    times1 = []
    times2 = []
    dts = []
    times_exceeded = []
    waiting_times = []
    for method, test_path in test_paths.items():
        with open(test_path['problems_path'], 'rb') as hh:
            problem_data = safe_pickle_load(hh)
        for i, path in enumerate(test_path['path']):
            hds = [p[0] for p in path]
            xy = positions[i][hds, :]
            travel = [0] + list((xy.diff(dim=0)**2).sum(dim=1).sqrt().numpy())
            tms = [p[3]/norm for p in path]
            tms1 = [problem_data['t_min'][i, h].item() for h in hds]
            tms2 = [problem_data['t_max'][i, h].item() for h in hds]
            dt0 = [problem_data['dt'][i, h].item() for h in hds]
            exceed = [(0 if (h == 0 or t1 < t-dt < t2) else ((t-dt - t1) if t-dt < t1 else (t-dt - t2)))
                      for h, t, t1, t2, dt in zip(hds, tms, tms1, tms2, dt0)]
            waiting_time = [0] + [0 if t2==0 else (t2-dt-t1)-d
                                  for t1,t2,dt,d in zip(tms[:-1],tms[1:],dt0[1:],travel[1:])]
            n = len(tms)
            methods.extend(n * [method])
            problems.extend(n * [i])
            heads.extend(hds)
            travels.extend(travel)
            times.extend(tms)
            times1.extend(tms1)
            times2.extend(tms2)
            dts.extend(dt0)
            times_exceeded.extend(exceed)
            waiting_times.extend(waiting_time)

    tt = dict(method=methods, env_id=problems, head=heads, travel=travels, time=times, t_min=times1, t_max=times2,
              time_exceeded=times_exceeded, waiting=waiting_times)
    try:
        tt = pd.DataFrame(tt)
    except:
        print({k:len(v) for k,v in tt.items()})
        raise
    return tt

def show_trajectories(res, n_trajectories=5, fontsize=15, cmap='Blues', axargs=None):
    # set axes
    default_axargs = dict(W=3, axsize=(6,4))
    if axargs is not None:
        for k,v in axargs.items():
            default_axargs[k] = v
    axs = Axes(n_trajectories, **default_axargs)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())

    # extract returns
    rets = sorted(res.ret.returns)
    ids = np.linspace(0, len(res)-1, n_trajectories).astype(int)
    rets = np.take(rets, ids)

    # choose trajectories and visualize
    for i, ret in enumerate(rets):
        r = res[res.ret == ret]
        episode = r.episode.returns[0]
        r = r[r.episode == episode]
        show_trajectory(res.x.returns, res.y.returns, res.k.returns,
                        axs[i], fontsize=fontsize, cmap=cmap, sm=sm)
        labels(axs[i], 'x', 'y', f'iter={res.iteration.returns[0]}, env={res.i_env.returns[0]}, return={res.ret.returns[0]:.2f}',
               fontsize=fontsize)

    plt.tight_layout()
    return axs

def show_trajectory(x, y, k, demands=None, vehicles=None, times=None, ax=None, fontsize=15,
                    annotate=None, color_per_car=True, cmap=None, sm=None, colorbar=True,
                    markersize=3):
    if ax is None:
        ax = Axes(1, 1)[0]
    if sm is None:
        if cmap is None:
            if color_per_car is None:
                color_per_car = vehicles is not None
            elif color_per_car:
                vehicles = np.cumsum(np.array(k) == 0)
                vehicles[-1] = vehicles[-2]
            cmap = 'cool' if not color_per_car else 'rainbow'
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())
    if annotate is None:
        annotate = len(x) <= 100

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    if not color_per_car:
        counts = np.arange(len(x) - 1)
        clim = (-5, counts[-1])
    else:
        counts = vehicles[:-1]
        clim = (counts[0], counts[-1])
    sm.set_array(counts)
    ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], counts, scale_units='xy', angles='xy', scale=1,
              color='k', cmap=cmap, clim=clim, width=0.0025, headwidth=10, headlength=12)
    if colorbar:
        plt.colorbar(sm, ax=ax)
    ax.plot(x[-1:], y[-1:], 'rs', markersize=7)
    ax.plot(x[:1], y[:1], 'g>', markersize=8)
    if demands is None:
        ax.plot(x, y, 'ko', markersize=markersize)
    else:
        ax.scatter(x, y, c=['g']+['y' if dem>0 else 'purple' for dem in demands[1:]],
                   marker='o', s=80, zorder=2)
    if annotate:
        annotations = [f'{i:d}:{k[i]:d}' for i in range(len(x))]
        if demands is not None:
            for i, demand in enumerate(demands):
                annotations[i] += f' ({demand:.2f})'
        if times is not None:
            for i, t in enumerate(times):
                annotations[i] += f' [{t:.1f}]'
        for i, annot in enumerate(annotations):
            ax.annotate(annot, (x[i], y[i]), fontsize=fontsize - 3)
    labels(ax, 'x', 'y', fontsize=fontsize)

    return ax

def show_trajectory_on_map(x, y, k, area, demands=None, vehicles=None, ax=None, fontsize=16,
                           color_per_car=True, cmap=None, sm=None, colorbar=False, markersize=25, alpha=0.5):
    if ax is None:
        ax = Axes(1, 1)[0]
    vehicles = np.cumsum(np.array(k) == 0)
    vehicles[-1] = vehicles[-2]
    if sm is None:
        if cmap is None:
            if color_per_car is None:
                color_per_car = vehicles is not None
            cmap = 'cool' if not color_per_car else 'rainbow'
        # if cmap == 'rainbow':
        #     import matplotlib.colors as mcolors
        #     def darken_cmap(cmap, scale=0.75):
        #         cmap = plt.cm.get_cmap(cmap)
        #         colors = cmap(np.arange(cmap.N))
        #         colors = (colors[:, :3] * scale).clip(0, 1)
        #         return mcolors.ListedColormap(colors)
        #     cmap = darken_cmap(cmap, scale=0.75)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    if not color_per_car:
        counts = np.arange(len(x) - 1)
        clim = (-5, counts[-1])
    else:
        counts = vehicles[:-1]
        clim = (counts[0], counts[-1])
    sm.set_array(counts)

    geolocations = [xy_to_geolocation(xx, yy, area) for xx, yy in zip(x, y)]
    lngs = [p[0] for p in geolocations]
    lats = [p[1] for p in geolocations]

    # customers
    geometry = [Point(lng, lat) for lng, lat in zip(lngs, lats)]
    geo_df = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)
    geo_df.plot(ax=ax, marker='.', markersize=markersize, color='k', label='customers')

    # depot
    depot_geom = [Point(lngs[0], lats[0])]
    depot_df = gpd.GeoDataFrame(geometry=depot_geom, crs="EPSG:4326").to_crs(epsg=3857)
    depot_df.plot(ax=ax, marker='s', markersize=6*markersize, color='k', label='depot')

    # routes
    # line = LineString(geometry)
    # line_df = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")
    # line_df = line_df.to_crs(epsg=3857)
    # line_df.plot(ax=ax, color=sm)
    coords = np.array([point.coords[0] for point in geo_df.geometry])
    ax.quiver(coords[:-1,0], coords[:-1,1], np.diff(coords[:, 0]), np.diff(coords[:, 1]), counts,
              scale_units='xy', angles='xy', scale=1, color='k', cmap=cmap, clim=clim,
              width=0.004, headwidth=10, headlength=12)

    # map
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=alpha)
    ax.axis('off')
    labels(ax, title=area, fontsize=fontsize)
    # ax.legend(fontsize=fontsize-3)

    if colorbar:
        plt.colorbar(sm, ax=ax)

    return ax

def xy_to_geolocation(x, y, city):
    center_map = dict(rio=(-22.908333, -43.196388), sao_paulo=(-23.533773, -46.625290))
    center = center_map[city]
    lat = y / 110574 + center[0]
    lng = x / (111320 * np.cos(np.deg2rad(center[0]))) + center[1]
    return lng, lat


def show_partial_trajectory(nodes, path_ids, demands=None, ax=None, fontsize=15, cmap=None, sm=None):
    if ax is None:
        ax = Axes(1, 1)[0]
    if sm is None:
        if cmap is None:
            cmap = 'cool'
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())

    x_all = nodes[:, 0]
    y_all = nodes[:, 1]
    x = np.array([nodes[i, 0] for i in path_ids])
    y = np.array([nodes[i, 1] for i in path_ids])

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    counts = np.arange(len(x) - 1)
    clim = (-5, counts[-1])
    sm.set_array(counts)
    ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], counts, scale_units='xy', angles='xy', scale=1,
              color='k', cmap=cmap, clim=clim, headwidth=6, headlength=8)
    plt.colorbar(sm, ax=ax)
    ax.plot(x[:1], y[:1], 'g>', markersize=16)
    if demands is None:
        ax.plot(x_all, y_all, 'yo', markersize=7)
    else:
        ax.scatter(x_all, y_all, c=['g']+['y' if dem>0 else 'purple' for dem in demands[1:]],
                   marker='o', s=80, zorder=2)
    annotations = [f'{i:d}' for i in range(len(x_all))]
    if demands is not None:
        for i, demand in enumerate(demands):
            annotations[i] += f' ({demand:.2f})'
    for i, annot in enumerate(annotations):
        ax.annotate(annot, (x_all[i], y_all[i]), fontsize=fontsize - 3)
    labels(ax, 'x', 'y', fontsize=fontsize)

    return ax


def display_solution(sol, problems, problem_id, area='sao_paulo', print_vehicles=True, axsize=(6, 5)):
    cost = solution_cost(sol, problems['distance_matrix'][problem_id])

    # get area, positions, and vehicles
    positions = problems['positions']
    pos = positions[problem_id][sol, :]
    n_cars = int(np.sum([k == 0 for k in sol[:-1]]))

    # plot
    axs = Axes(1, 1, axsize)
    a = 0

    if area is None:
        show_trajectory(pos[:, 0], pos[:, 1], sol, ax=axs[a])
    else:
        show_trajectory_on_map(pos[:, 0], pos[:, 1], sol, area, ax=axs[a], alpha=0.6, cmap='Dark2')
    if print_vehicles:
        axs[a].set_title(f'vehicles={n_cars:d}, cost={cost:.0f}', fontsize=16)
    else:
        axs[a].set_title(f'cost={cost:.0f}', fontsize=16)
    a += 1

    plt.tight_layout()
    return axs


def display_solutions(rr, sol0, sol1, problems, problem_id, method0, method1, time, methods_map=None, area='sao_paulo',
                      print_vehicles=True, axsize=(6, 5)):
    # align solutions
    sol1 = routes_matching.optimize_sequence_order(sol0, sol1)

    # update methods names
    method0 = methods_map[method0]
    method1 = methods_map[method1]

    # get recorded costs
    cost0 = rr[(rr.total_runtime == time) & (rr.method == method0) & (rr.problem_id == problem_id)].cost.values[0]
    cost1 = rr[(rr.total_runtime == time) & (rr.method == method1) & (rr.problem_id == problem_id)].cost.values[0]

    # calculate costs from solutions and distances - verify consistency
    cost0b = solution_cost(sol0, problems['distance_matrix'][problem_id])
    cost1b = solution_cost(sol1, problems['distance_matrix'][problem_id])
    inconsistency = False
    if (np.abs(cost0b - cost0) / cost0) > 1e-4:
        print(f'Insconsistent cost for {method0}: recorded={cost0}, calculated={cost0b}')
        inconsistency = True
    if (np.abs(cost1b - cost1) / cost1) > 1e-4:
        print(f'Insconsistent cost for {method1}: recorded={cost1}, calculated={cost1b}')
        inconsistency = True

    # get area, positions, and vehicles
    positions = problems['positions']
    pos0 = positions[problem_id][sol0, :]
    pos1 = positions[problem_id][sol1, :]
    n_cars0 = int(np.sum([k == 0 for k in sol0[:-1]]))
    n_cars1 = int(np.sum([k == 0 for k in sol1[:-1]]))

    # plot
    axs = Axes(2, 2, axsize)
    a = 0

    if area is None:
        show_trajectory(pos0[:, 0], pos0[:, 1], sol0, ax=axs[a])
    else:
        show_trajectory_on_map(pos0[:, 0], pos0[:, 1], sol0, area, ax=axs[a], alpha=0.6, cmap='Dark2')
    if print_vehicles:
        axs[a].set_title(f'{method0}: vehicles={n_cars0:d}, cost={cost0:.0f}', fontsize=16)
    else:
        axs[a].set_title(f'{method0}: cost={cost0:.0f}', fontsize=16)
    a += 1

    if area is None:
        show_trajectory(pos1[:, 0], pos1[:, 1], sol1, ax=axs[a])
    else:
        show_trajectory_on_map(pos1[:, 0], pos1[:, 1], sol1, area, ax=axs[a], alpha=0.6, cmap='Dark2')
    if print_vehicles:
        axs[a].set_title(f'{method1}: vehicles={n_cars1:d}, cost={cost1:.0f}', fontsize=16)  # [SP2Rio-100-{time}s]
    else:
        axs[a].set_title(f'{method1}: cost={cost1:.0f}', fontsize=16)  # [SP2Rio-100-{time}s]
    a += 1

    plt.tight_layout()
    return axs

def plot_frontier(frontier, pos, state, scale=25, ax=None, fontsize=15):
    if ax is None:
        ax = Axes(1, 1)[0]

    if len(state) > 3:
        head, demand, capacity, _, _ = state
    else:
        head, demand, capacity = state
    capacity = capacity[0]
    ids = list(frontier.keys())

    # plot current head
    ax.scatter(*pos[head], s=2*scale, marker='s', c='black')

    # plot depot
    ax.scatter(*pos[0], s=2*scale, marker='s', c='orange')

    # plot invalid actions
    fulfilled = [i for i in range(len(pos)) if i != head and i not in ids and demand[i] == 0]
    infeasible = [i for i in range(len(pos)) if i != head and i not in ids and demand[i] != 0]
    ax.scatter(pos[fulfilled, 0], pos[fulfilled, 1], s=scale, c='lightgreen')
    ax.scatter(pos[infeasible, 0], pos[infeasible, 1], s=scale, c='darkred')

    # plot valid actions
    vals = [v[0] for v in frontier.values()]
    selected = [v[1] for v in frontier.values()]
    col = ['m' if sel else 'b' for sel in selected]
    x = pos[ids, 0]
    y = pos[ids, 1]
    s = [-1/(v+0.1) for v in vals]
    s = [ss*scale/np.mean(s) for ss in s]
    hh = ax.scatter(x, y, s=s, c=col)

    for i, v in enumerate(vals):
        ax.annotate(f'{v:.2f}', (x[i], y[i]), ha='center', fontsize=fontsize-3)

    infeasible_demands = [dem for i, dem in enumerate(demand) if i in infeasible]
    for i, dem in zip(infeasible, infeasible_demands):
        ax.annotate(f'd:{dem:.2f}', (pos[i, 0], pos[i, 1]), ha='left', fontsize=fontsize-3)
    ax.annotate(f'C:{capacity:.2f}', (pos[head, 0], pos[head, 1]), ha='left', fontsize=fontsize-3)

    labels(ax, 'x', 'y', fontsize=fontsize)

    legend_elements = [
        Line2D([0], [0], color='black', marker='s', linestyle='None', label='current'),
        Line2D([0], [0], color='magenta', marker='o', linestyle='None', label='selected'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', label='nonselected'),
        Line2D([0], [0], color='lightgreen', marker='o', linestyle='None', label='fulfilled'),
        Line2D([0], [0], color='darkred', marker='o', linestyle='None', label='infeasible'),
        Line2D([0], [0], color='orange', marker='s', linestyle='None', label='depot'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=fontsize-3)

    return ax

def show_move(path_logs, problem, step=None, problems_info=None,
              equal_axes=True, unnormalize_ret=True, normalization=None,
              method='rlopt', detailed_times=False, axs=None, a0=0, axs_kwargs=None, **kwargs):

    pos = path_logs['positions'][problem]
    ret = path_logs['net_return'][problem] if 'net_return' in path_logs else path_logs['return'][problem]
    if unnormalize_ret:
        if normalization is None:
            problem_size = len(pos) - 1
            normalization = 1 / np.sqrt(3.14 * problem_size)
        ret = - ret / normalization
    else:
        normalization = 1
    path = path_logs['path'][problem]
    path_id = [x[0] for x in path]
    path_vehicles = None
    if len(path[0]) > 3:
        path_vehicles = [p[4] for p in path]
    path_demands = None
    path_times = None
    if detailed_times:
        path_demands = [path[0][2][0]] + [b[2][0]-a[2][0] for a,b in zip(path[:-1],path[1:])]
        if len(path[0]) > 3:
            path_times = [p[3] / normalization for p in path]
    frontiers = path_logs['frontiers'][problem]
    xx = pos[path_id, 0]
    yy = pos[path_id, 1]

    is_with_baselines = False
    if problems_info is not None:
        if 'baseline_solutions_lists' in problems_info or 'baseline_solutions' in problems_info:
            is_with_baselines = True
    show_frontier = step is not None
    if axs is None:
        axs_default_kwargs = dict(N = 1 + is_with_baselines + show_frontier, W=3, axsize=(7,6))
        if axs_kwargs is not None:
            for k,v in axs_kwargs.items():
                axs_default_kwargs[k] = v
        axs = Axes(**axs_default_kwargs)
    a = a0

    show_trajectory(xx, yy, path_id, demands=path_demands, vehicles=path_vehicles, times=path_times,
                    ax=axs[a], **kwargs)
    if equal_axes:
        axs[a].set_aspect('equal')
    axs[a].set_title(f'[{method}] env #{problem:d}: {ret:.3f}', fontsize=15)
    a += 1

    if is_with_baselines:
        base_ret = problems_info['baseline_cost_unnormalized'][problem]
        base_path_id = problems_info['baseline_solutions_lists'][problem]
        truck_id = np.cumsum(np.array(base_path_id)==0)
        truck_id[-1] = truck_id[-2]
        base_xx = pos[base_path_id, 0]
        base_yy = pos[base_path_id, 1]
        show_trajectory(base_xx, base_yy, base_path_id, vehicles=truck_id, ax=axs[a], **kwargs)
        if equal_axes:
            axs[a].set_aspect('equal')
        axs[a].set_title(f'[cuopt] env #{problem:d} baseline: {base_ret:.3f}', fontsize=15)
        a += 1

    if show_frontier:
        tt = step
        # print(path[tt])
        head = path[tt][0]
        frontier = frontiers[tt][path[tt]]
        state = path[tt]
        plot_frontier(frontier, pos, state, scale=50, ax=axs[a])
        if equal_axes:
            axs[a].set_aspect('equal')
        axs[a].set_title(f'step={tt:d}, node={head:d}', fontsize=15)
        a += 1

    plt.tight_layout()
    return axs


def cuopt_acceptance(path_logs, to_plot=False, time_rounding=0):
    acceptance = [int(x) for snapshot in path_logs['cuopt_acceptance'] for x in snapshot]
    strong_acceptance = [int(x) for snapshot in path_logs['cuopt_strong_acceptance'] for x in snapshot]
    times = [x for snapshot_times in path_logs['cuopt_times'] for x in snapshot_times]
    problem_id = [len(snpshot_times) * [i] for i, snpshot_times in enumerate(path_logs['cuopt_times'])]
    problem_id = [i for ids in problem_id for i in ids]

    aa = dict(problem_id=problem_id, time=times, accepted=acceptance, strongly_accepted=strong_acceptance)
    try:
        aa = pd.DataFrame(aa)
    except:
        warnings.warn('Failed to construct a dataframe of cuopt acceptance results.')
        print({k: len(v) for k,v in aa.items()})
        return pd.DataFrame()

    if to_plot:
        plot_cuopt_acceptance(aa, time_rounding)

    return aa

def summarize_cuopt_acceptance(all_path_logs, to_plot=False, time_rounding=0):
    aa = pd.DataFrame()
    for k, v in all_path_logs.items():
        df = cuopt_acceptance(v, to_plot=False)
        df['method'] = len(df) * [k]
        aa = pd.concat((aa, df))

    if to_plot:
        plot_cuopt_acceptance(aa, time_rounding, multi=True)

    return aa

def plot_cuopt_acceptance(aa, time_rounding=0, multi=False, axs=None, a0=0, start_from_0=True):
    if axs is None:
        axs = Axes(2, 2, (3.5, 3))

    if time_rounding > 0:
        aa['time_rounded'] = [time_rounding * (tt // time_rounding) for tt in aa.time]
    else:
        aa['time_rounded'] = aa['time']

    if multi:
        sns.lineplot(data=aa, y='accepted', x='time_rounded', errorbar=('ci', 95),
                     estimator=lambda x: 100*np.mean(x), hue='method', ax=axs[a0])
        sns.lineplot(data=aa, y='strongly_accepted', x='time_rounded', errorbar=('ci', 95),
                     estimator=lambda x: 100 * np.mean(x), hue='method', ax=axs[a0+1])
    else:
        sns.lineplot(data=aa, y='accepted', x='time_rounded', errorbar=('ci', 95),
                     estimator=lambda x: 100*np.mean(x), ax=axs[a0])
        sns.lineplot(data=aa, y='strongly_accepted', x='time_rounded', errorbar=('ci', 95),
                     estimator=lambda x: 100*np.mean(x), ax=axs[a0+1])

    if start_from_0:
        axs[a0].set_xlim((0, None))
        axs[a0+1].set_xlim((0, None))

    axs.labs(a0, 'time [s]', 'P(accept) [%]', fontsize=15)
    axs.labs(a0+1, 'time [s]', 'P(strong_accept) [%]', fontsize=15)

    plt.tight_layout()
    return axs

def analyze_test_solutions(returns, vehicles, ref_rets, ref_vehicles):
    '''
    returns[method][i_problem] = return
    solutions[method][i_problem] = list of nodes
    ref_rets[i_problem] = return
    ref_cars[i_problem] = list of nodes
    '''

    methods = []
    problem_ids = []
    ret = []
    base_ret = []
    cars = []
    base_cars = []

    for method in returns:
        for ind, (curr_ret, curr_cars, ref_ret) in enumerate(zip(
                returns[method], vehicles[method], ref_rets)):
            problem_ids.append(ind)
            methods.append(method)
            ret.append(curr_ret)
            base_ret.append(ref_ret)
            cars.append(curr_cars)
            base_cars.append(ref_vehicles[ind])

    rr = pd.DataFrame(dict(
        method=methods,
        problem_id=problem_ids,
        baseline_cars=base_cars,
        baseline_ret=base_ret,
        cars=cars,
        ret=ret,
    ))

    rr['cost_gap'] = rr.ret / rr.baseline_ret
    rr['cars_gap'] = rr.cars - rr.baseline_cars
    rr['vehicle gap'] = [f'{gap:.0f}' for gap in rr.cars_gap]

    axs = Axes(5, 3)
    a = 0

    qplot(rr, 'cost_gap', 'problem', 'method', ax=axs[a])
    a += 1

    qplot(rr, 'cars_gap', 'problem', 'method', ax=axs[a])
    axs.labs(a, ylab='vehicle_gap', fontsize=15)
    a += 1

    hue_order = sorted(pd.unique(rr['vehicle gap']), reverse=True)
    sns.histplot(data=rr, x='method', hue='vehicle gap', hue_order=hue_order,
                 ax=axs[a], multiple='stack')  # , stat='percent')
    axs.labs(a, title=f'no_gap(rlopt)={100*(rr[rr.method=="rlopt"].cars_gap==0).mean():.1f}%', fontsize=15)
    a += 1

    if (rr.cars_gap==0).any():
        qplot(rr[rr.cars_gap==0], 'cost_gap', 'problem', 'method', ax=axs[a])
        axs.labs(a, title='only vehicle-competitive solutions', fontsize=15)
    a += 1

    if (rr.cars_gap>0).any():
        qplot(rr[rr.cars_gap>0], 'cost_gap', 'problem', 'method', ax=axs[a])
        axs.labs(a, title='only vehicle-suboptimal solutions', fontsize=15)
    a += 1

    plt.tight_layout()
    return axs

def display_game_tree(nodes, ax=None, embedding='graphviz', colormap='cool', validate=True):
    # nodes = tree.reached = {state: node} for all reached nodes in game

    if ax is None:
        ax = Axes(1, 1, (8, 7))[0]
    cmap = plt.get_cmap(colormap)

    # validate graph structure
    if validate:
        for state, node in nodes.items():
            for child in node.children.values():
                if state not in [par.hashable_state for par in child.parents]:
                    warnings.warn(f'Node B (depth {node.depth}) points to child C (depth {child.depth}), '
                                  f'which does not point back.')
            if node.parents is not None:
                for par in node.parents:
                    if state not in [child.hashable_state for child in par.children.values()]:
                        warnings.warn(f'Node B (depth {node.depth}) points to parent A (depth {par.depth}), '
                                      f'which does not point back.')

    # create graph
    G = nx.DiGraph()
    max_depth = 0
    for state, node in nodes.items():
        max_depth = max(max_depth, node.depth)
        for child in node.children.values():
            G.add_edge(state, child.hashable_state)

    # assign colors
    colors = []
    for state in G.nodes():
        if nodes[state].parents is None:
            colors.append('lightgreen')
        elif nodes[state].terminal:
            colors.append('red')
        else:
            colors.append(mpl.colors.rgb2hex(cmap(nodes[state].depth/max_depth)))

    # create 2D embeddings
    if embedding == 'graphviz':
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
    elif embedding == 'planar':
        # this one is pretty shitty
        GU = G.to_undirected()
        is_planar, embedding = nx.check_planarity(GU)
        pos = nx.combinatorial_embedding_to_pos(embedding)
    elif embedding == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        raise ValueError(embedding)

    nx.draw(G, pos, arrows=True, node_size=50, node_color=colors, ax=ax)
    return ax

def edge_coverage_analysis(solutions, verbose=1):
    if verbose >= 1:
        print('\nSolutions coverage analysis:')
        print(f'Average number of solutions: {np.mean([len(prob_sols) for prob_sols in solutions]):.1f}')
    problem_size = len(np.unique(solutions[0][0]))
    n_edges = []
    edge_cover = []
    all_edge_counts = []
    mean_shared_edges = []
    for prob_sols in solutions:
        edge_count = np.zeros((problem_size, problem_size))
        for sol in prob_sols:
            for node1, node2 in zip(sol[:-1], sol[1:]):
                if node1 > node2:
                    edge_count[node1, node2] += 1
                else:
                    edge_count[node2, node1] += 1
        edge_count = np.array([edge_count[i,j] for i in range(problem_size) for j in range(i)])
        n_edges.append(len(edge_count))
        edge_cover.append(np.sum(edge_count > 0))
        all_edge_counts = np.concatenate((all_edge_counts, edge_count))
        # calculate average edge sharing per pair of solutions
        #  (note: if an edge appears in n solutions, then it accounts for n-choose-2 edge-sharings.)
        edge_hist = collections.Counter(edge_count)
        total_shared_edges = np.sum([count*(visits*(visits-1)/2) for visits, count in edge_hist.items()])
        mean_shared_edges.append(total_shared_edges / (len(prob_sols)*(len(prob_sols)-1)/2) / problem_size)
    n_edges = np.array(n_edges)
    edge_cover = np.array(edge_cover)
    rel_cover = edge_cover / n_edges
    if verbose:
        print(f'Average explored edges: {np.mean(edge_cover):.0f}/{np.mean(n_edges):.0f} = '
              f'{100*np.mean(rel_cover):.3f}% (+-{100*np.std(rel_cover)/np.sqrt(len(rel_cover)):.3f}%)')
    # print('Histogram of edge visits:')
    # edge_hist = collections.Counter(all_edge_counts)
    # print(edge_hist)
    mean_shared_edges = np.mean(mean_shared_edges)
    if verbose:
        print(f'Percent of shared-edges (averaged over problems, and over solution-pairs per problem):'
              f' {100*mean_shared_edges:.2f}%')
    return np.mean(rel_cover), mean_shared_edges

def solution_cost(sol, positions, Lp=2):
    if positions.shape[1] > 2:
        # interpret positions as distances
        distances = positions
        return np.sum([distances[i1,i2] for i1, i2 in zip(sol[:-1], sol[1:])])
    return np.sum([(np.linalg.norm(positions[i2] - positions[i1], ord=Lp))
                   for i1, i2 in zip(sol[:-1], sol[1:])])

def add_col_all_success(r, group_cols=('total_runtime', 'problem_id')):
    def add_col(d):
        d['all_success'] = d.success.all()
        return d
    return r.groupby(list(group_cols), group_keys=False).apply(add_col)

def injection_methods_names_update(rr, pp=None, methods_map=None, methods_list=None):
    if methods_map is not None:
        if methods_list is None:
            methods_list = list(methods_map.keys())
        rr['method0'] = rr.method
        rr = rr[rr.method.isin(methods_list)].copy()
        rr['method'] = rr.method0.transform(lambda m: methods_map[m] if m in methods_map else m)
        if pp is not None:
            pp['method0'] = pp.method
            pp = pp[pp.method.isin(methods_list)].copy()
            pp['method'] = pp.method0.transform(lambda m: methods_map[m] if m in methods_map else m)
    return rr, pp

def injection_post_process(rr, pp=None, methods_map=None, methods_list=None, baseline=None,
                           max_accepted_display=5, time_resolution=4):
    # clone
    rr = rr.copy()
    if pp is not None:
        pp = pp.copy()

    # update methods names
    if methods_map is not None:
        rr, pp = injection_methods_names_update(rr, pp, methods_map, methods_list)

    # make sure problem-list is the same for all the methods
    len_rr = len(rr)
    n_methods = rr.method.nunique()
    method_set = set(list(rr.method.unique()))
    def foo(d):
        if len(d) == n_methods:
            return d
        elif d.method.nunique() == n_methods:
            print(f'[{d.total_runtime.values[0]}, {d.problem_id.values[0]}]\t{len(d)} != {n_methods}')
            return d.drop_duplicates(subset='method', keep='first')
        else:
            missing_methods = method_set - set(list(d.method.unique()))
            print(f'[{d.total_runtime.values[0]}, {d.problem_id.values[0]}]\t{len(d)} != {n_methods};\tmissing:', missing_methods)
            return pd.DataFrame()
    rr = rr.groupby(['total_runtime','problem_id'], group_keys=False).apply(foo)
        # lambda d: d if len(d)==n_methods else pd.DataFrame()).reset_index(drop=True)
    if len(rr) < len_rr:
        warnings.warn(f'Removed {100*(len_rr-len(rr))/len_rr:.1f}% of the runs '
                      f'to make sure each remaining problem has data from all methods.')

    # add costs transformations
    for ycol in ('vehicles', 'cost'):
        rr_best = rr.copy()
        if rr_best.total_runtime.nunique() > 1:
            rr_best.loc[rr_best.total_runtime == rr_best.total_runtime.min(), ycol] = np.inf
        rr[f'best_{ycol}'] = rr_best[rr.success].groupby('problem_id')[ycol].transform('min')
        rr[f'{ycol}_suboptimality_diff [%]'] = 100 * (rr[ycol].values - rr[f'best_{ycol}'].values) / rr[
            f'best_{ycol}'].mean()
        rr[f'{ycol}_suboptimality_ratio [%]'] = 100 * (rr[ycol].values / rr[f'best_{ycol}'].values - 1)
        if baseline is not None:
            rr[f'{ycol}_vs_{baseline}_ratio [%]'] = 0
            for method in pd.unique(rr.method):
                rr.loc[rr.method == method, f'{ycol}_vs_{baseline}_ratio [%]'] = 100 * (
                            rr.loc[rr.method == method, ycol].values / rr.loc[rr.method == baseline, ycol].values - 1)
    rr['n_accepted'] = [str(acc) if acc < max_accepted_display else f'{max_accepted_display:d}+' for acc in rr.accepted]

    if pp is not None:
        pp['time_rounded'] = [time_resolution * (t // time_resolution) for t in pp.time]
        pp['opt_cost'] = pp.groupby('problem_id')['best_cost'].transform('min')
        pp['cost_suboptimality [%]'] = 100 * (pp.best_cost.values / pp.opt_cost.values - 1)

    return rr, pp

def project_baseline_on_all_methods(rr, baseline='CuOpt', y_col='cost_suboptimality_ratio [%]', B=100, verbose=False,
                                    filter_nas=True, avoid_non_decreasing=True, interpolation_kwargs=None):
    if interpolation_kwargs is None:
        interpolation_kwargs = {}

    def interpolate_x_for_y(x, y, yp, clip_back=False, extrapolate_forward=False):
        if yp > y[0]:
            return x[0] if clip_back else None
        if yp < y[-1]:
            if extrapolate_forward:
                return x[-1] + (yp - y[-1]) * (x[-1] - x[-2]) / (y[-1] - y[-2]) if y[-1] - y[-2] != 0 else None
            else:
                return None
        for i in range(1, len(y))[::-1]:
            if y[i - 1] >= yp >= y[i]:
                if y[i] - y[i - 1] == 0:
                    return x[i - 1]
                return x[i - 1] + (x[i] - x[i - 1]) * ((yp - y[i - 1]) / (y[i] - y[i - 1]))
        raise RuntimeError

    def agg_problems(rr):
        rr[y_col] = rr[y_col].mean()
        return rr.head(1)

    def align_methods(ss, m0, m1):
        s0 = ss[ss.method == m0].copy()
        s1 = ss[ss.method == m1].copy()
        s0['projected_y'] = [interpolate_x_for_y(s0.total_runtime.values, s0[y_col].values, yp, **interpolation_kwargs)
                             for yp in s1[y_col].values]
        return s0

    def project_bs_sample(rr, baseline):
        # generate BS sample
        t0 = rr.total_runtime.values[0]
        r0 = rr[(rr.method == baseline) & (rr.total_runtime == t0)]
        problem_ids = r0.problem_id.unique()
        if B > 1:
            problem_ids = np.random.choice(problem_ids, len(r0), replace=True)
        rr = pd.concat([rr[rr.problem_id==pid] for pid in problem_ids], ignore_index=True)

        # project times for BS sample
        ss = rr.groupby(['method', 'total_runtime'], sort=False).apply(
            lambda r: agg_problems(r)).reset_index(drop=True)
        if avoid_non_decreasing:
            for m in ss.method.unique():
                if np.any(np.diff(ss.loc[ss.method==m, y_col].values) > 0):
                    if verbose:
                        print(f'non-decreasing {y_col} for {m}')
                    return pd.DataFrame()
        ss = pd.concat([align_methods(ss, m0, baseline) for m0 in pd.unique(ss.method)], ignore_index=True)
        return ss

    def project_with_bs(rr):
        projections = [project_bs_sample(rr, baseline) for _ in range(B)]
        empties = np.sum([len(p)==0 for p in projections])
        if empties:
            print(f'{empties}/{len(projections)} invalid BS samples were filtered out.')
        return pd.concat([project_bs_sample(rr, baseline) for _ in range(B)]).reset_index(drop=True)

    ss = project_with_bs(rr)

    if filter_nas:
        ss['method_and_time'] = list(zip(ss.method.values, ss.total_runtime.values))
        bad_entries = ss[ss.projected_y.isna()].method_and_time.unique()
        ss = ss[~ss.method_and_time.isin(bad_entries)]

    return ss


def time_to_cost(rr, cost_range=(0, 5.1, 0.1), time='total_runtime', cost='cost_suboptimality_ratio [%]',
                 filter='success', interpolate=False, const_cols=('method', 'problem_id')):
    def group_time_to_cost(dd):
        if filter:
            dd = dd[dd[filter]]
            if len(dd) == 0:
                return pd.DataFrame()
        data_costs = dd[cost].values
        data_times = dd[time].values
        costs = np.arange(*cost_range)
        time_to_cost = []
        fast_reach = []
        for c in costs:
            k = np.where(data_costs <= c)[0]
            if len(k) == 0:
                fast_reach.append(False)
                time_to_cost.append(np.nan)
            else:
                k = k[0]
                if k == 0:
                    fast_reach.append(True)
                    time_to_cost.append(data_times[k])
                elif interpolate:
                    fast_reach.append(False)
                    time_to_cost.append(
                        data_times[k] + (c - data_costs[k]) * (data_times[k-1] - data_times[k]) / (data_costs[k-1]-data_costs[k]))
                else:
                    fast_reach.append(False)
                    time_to_cost.append(data_times[k])
            # d = dd[dd[cost] <= c]
            # if len(d) > 0 and (len(d) < len(dd) or not filter_shortest_time):
            #     if interpolate:
            #         time_to_cost.append(np.interp(c, dd[cost], dd[time]))
            #         import pdb
            #         pdb.set_trace()
            #     else:
            #         time_to_cost.append(d[time].values[0])
            # else:
            #     time_to_cost.append(np.nan)
        dct = dict(
            cost = costs,
            time_to_cost = time_to_cost,
            success = ~np.isnan(time_to_cost),
            fast_reach = fast_reach,
        )
        for col in const_cols:
            dct[col] = dd[col].values[0]
        return pd.DataFrame(dct)

    return rr.groupby(list(const_cols)).apply(group_time_to_cost).reset_index(drop=True)

def plot_time_to_cost(tt, ax=None, hue='method', min_valid_ratio=0.0, extrapolate=True, median=False,
                      offsets=((0, -10), (-7, 0), (0, 10)), lab_freq=1, lab_offset=0.1):
    if ax is None:
        ax = Axes(1, 1)[0]

    tt = tt.groupby(['method', 'cost'], group_keys=False).apply(
        lambda d: d.assign(success_rate=d.success.mean())).reset_index(drop=True)

    # filtering one method due to failure of another skews the data significantly
    # tt = tt.groupby(['cost', 'problem_id'], group_keys=False).apply(
    #     lambda d: d.assign(all_methods_valid=d.success.all()))
    # tt = tt[tt.all_methods_valid]

    if not extrapolate:
        if median:
            tt = tt.groupby(['method', 'cost'], group_keys=False).apply(
                lambda d: d.assign(extrapolation=d.fast_reach.mean() > 0.5))
            tt = tt[~tt.extrapolation].reset_index(drop=True)
        else:
            warnings.warn('filtering clipped times causes bias in data.')
            tt = tt.groupby(['cost', 'problem_id'], group_keys=False).apply(
                lambda d: d.assign(extrapolation=d.fast_reach.any()))
            tt = tt[~tt.extrapolation].reset_index(drop=True)

    new_ax = sns.lineplot(data=tt[tt.success_rate >= min_valid_ratio], x='cost', y='time_to_cost', hue=hue, ax=ax,
                          estimator='median' if median else 'mean')
    ax.invert_xaxis()
    if lab_freq > 0:
        valid_samples = tt.groupby(['cost', hue]).apply(lambda d: d.success.mean())
        hue_order = new_ax.get_legend_handles_labels()[1]
        colors = dict(zip(hue_order, sns.color_palette(n_colors=tt[hue].nunique())))
        xys = dict(zip(hue_order, offsets))
        for (cost, method), success_rate in valid_samples.items():
            if np.abs(cost % lab_freq - lab_offset) < 1e-5:
                ax.annotate(f"{100 * success_rate:.0f}%",
                            (cost, tt[(tt['cost'] == cost) & (tt.method == method)]['time_to_cost'].mean()),
                            textcoords="offset points", xytext=xys[method], ha='center', color=colors[method])
    return ax

def get_time_saved(tt, baseline, main_method, median=False):
    '''
    input cols: method, cost, time_to_cost, problem_id
    calc mean time_to_cost(method, cost) over problem_ids.
    calc time_saved(cost) between methods.
    output cols: cost, time_saved.
    '''
    tt = tt[tt.method.isin([baseline, main_method])]

    def get_successful_problems(d):
        d['all_success'] = d.success.all()
        return d

    tt = tt.groupby(['problem_id', 'cost'], group_keys=False).apply(get_successful_problems).reset_index(drop=True)

    def filter_successful_problems(d, thresh=0.5):
        if d.all_success.mean() < thresh:
            return pd.DataFrame()
        d['valid_problems_per_cost'] = d.all_success.mean()
        return d[d.all_success]

    tt = tt.groupby(['cost'], group_keys=False).apply(filter_successful_problems).reset_index(drop=True)
    if not len(tt):
        return tt

    def get_mean_time_per_cost(d):
        if median and d.time_to_cost.median() == d.time_to_cost.min():
            return pd.DataFrame()
        d['time_to_cost'] = d.time_to_cost.apply('median' if median else 'mean')
        return d.head(1)

    tt = tt.groupby(['method', 'cost'], group_keys=False).apply(get_mean_time_per_cost).reset_index(drop=True)

    def calc_time_saved(d):
        assert len(d) in (1, 2)
        if len(d) == 1:
            return pd.DataFrame()
        t0 = d[d.method == baseline].time_to_cost.values[0]
        t1 = d[d.method == main_method].time_to_cost.values[0]
        d['time_saved'] = 100 * (t0 - t1) / t0
        return d.head(1)

    tt = tt.groupby(['cost'], group_keys=False).apply(calc_time_saved).reset_index(drop=True)
    return tt
