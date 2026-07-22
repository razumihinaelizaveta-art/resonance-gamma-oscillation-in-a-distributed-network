import numpy as np
def distances(
        source_coords, 
        target_coords, 
        Nx, Ny, 
        cyclic = True):

    #Convert inputs
    source = np.asarray(source_coords)
    target = np.asarray(target_coords)

    #Validate inputs 
    for name, coords in [("source_coords", source),
                         ("target_coords", target)]:

        if coords.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array, got {coords.ndim}D."
            )

        if coords.shape[1] != 2:
            raise ValueError(
                f"{name} must have shape (N, 2), got {coords.shape}."
            )

        if coords.shape[0] == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

    
    #Distance calculation
    dx = np.abs(source[:, None, 0] - target[None, :, 0])
    dy = np.abs(source[:, None, 1] - target[None, :, 1])

    if cyclic:
        dx = np.minimum(dx, Nx - dx)
        dy = np.minimum(dy, Ny - dy)

    distances = np.sqrt(dx**2 + dy**2)

    #Output formation
    dtype = [('i', np.int64), ('j', np.int64), ('distance', np.float64)]
    output = np.empty(distances.shape, dtype=dtype)

   
    i_idx, j_idx = np.indices(distances.shape, dtype=np.int64)
    output['i'] = i_idx
    output['j'] = j_idx
    output['distance'] = distances

    return output.flatten()
    
    
def chack_requred_names(desc:dic,names:list,host:str):
    for name in names:
        if not name in desc:
            raise RuntimeError(f"There is no {name} in the description for {host} connectivity")
    return
def connectivity(conname:str,desc:dict,source_pos:np.array,target_pos:np.array,L:float,H:float,cbc:bool=True,selfid:bool=False)->tuple:
    d = distances(source_pos,target_pos,L,H,cbc)
    geometry = desc['geometry']
    if   geometry['type'] == 'random':
        chack_requred_names(geometry,['probability'],conname+': geometry')
        p = geometry['probability']
        connectivity = [
            [pre,post,_d_]
            for pre,post,_d_ in d
            if (pre != post or selfid) and rnd.rand() < p]
    elif geometry['type'] == 'lognormal':
        from scipy.stats import lognorm
        chack_requred_names(geometry,['sigma','mu'],conname+': geometry')
        sigma, mu = geometry['sigma'], geometry['mu']
        # PDF value at this distance
        pdf_val = lognorm.pdf(np.array([_d_ for _,_,_d_ in d]), s=sigma, scale=np.exp(mu))
        # Peak PDF value (at distance = e^mu)
        peak_pdf = lognorm.pdf(np.exp(mu), s=sigma, scale=np.exp(mu))
        # Normalized probability in [0, 1]
        prob = [ min(pv / peak_pdf, 1.0) for pv in pdf_val ]
        
        # add connectivity based on Bernoulli trial 
        connectivity = [
            [pre,post,_d_]
            for (pre,post,_d_),p in zip(d,prob)
            if np.random.random() < p and (pre != post or selfid)
        ]
    elif geometry['type'] == 'exponential':
        chack_requred_names(geometry,['max','sigma'],conname+': geometry')
        pmax,psigma = geometry['max'],geometry['sigma']
        prob = pmax*np.exp(-d/psigma)
        connectivity = [
            [pre,post,_d_]
            for (pre,post,_d_),p in zip(d,prob)
            if np.random.random() < p and (pre != post or selfid)
        ]
    else:
        raise RuntimeError(f"{conname}: Unknown type of connectivity!")
        
    # remove distances as they mess up computation of conductance and delays 
    del d
    conductance = desc['conductance']
    if conductance['type'] == 'random':
        chack_requred_names(conductance,['min','max'],conname+': conductance')
        gmin,gmax = conductance['min'],conductance['max']
        connectivity = [
            [pre,post,d, gmin+(gmax-gmin)*np.random.random()]
            for pre,post,d in connectivity
        ]
    elif conductance['type'] == 'exponential':
        chack_requred_names(conductance,['max','sigma'],conname+': conductance')
        gmax,gsigma = conductance['max'],conductance['sigma']
        connectivity = [
            [pre,post,d, gmax*np.exp(-d/gsigma)]
            for pre,post,d in connectivity
        ]
    elif conductance['type'] == 'gaussian':
        chack_requred_names(conductance,['max','sigma'],conname+': conductance')
        gmax,gsigma = conductance['max'],conductance['sigma']
        connectivity = [
            [pre,post,d, gmax*np.exp(-(d/gsigma)**2)]
            for pre,post,d in connectivity
        ]
    elif conductance['type'] == 'linear':
        chack_requred_names(conductance,['offset','slope'],conname+': conductance')
        ga,gk = conductance['offset'],conductance['slope']
        connectivity = [
            [pre,post,d, ga+gk*d]
            for pre,post,d in connectivity
        ]
    else:
        raise RuntimeError(f"{conname}:Unknown type of conductance!")    
    delay = desc['delay']
    if delay['type'] == 'linear':
        chack_requred_names(delay,['min','k'],conname+': delay')
        dmin, dk = delay['min'],delay['k']
        connectivity = [
            [pre,post,d,g,dmin+d/dk]
            for pre,post,d,g in connectivity
        ]
    else:
        raise RuntimeError(f"{conname}:Unknown type of delay!")
    return connectivity
